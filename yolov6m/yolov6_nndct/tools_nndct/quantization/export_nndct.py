# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
import time
import sys
import os
import torch
import torch.nn as nn
import onnx
import copy
from pathlib import Path
from tqdm import tqdm
import itertools

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
YOLOV6_PATH = (Path(ROOT).parent / 'yolov6').as_posix()
if YOLOV6_PATH not in sys.path:
    sys.path.append(YOLOV6_PATH)
from yolov6_nndct.core.engine import *
from yolov6_nndct.layers.common import *
from yolov6_nndct.models.effidehead import *
from yolov6_nndct.models.yolo import *
from yolov6_nndct.models.reppan import *
from yolov6_nndct.utils.checkpoint import *

from yolov6.models.yolo import build_model
from yolov6.models.effidehead import Detect
from yolov6.layers.common import *
from yolov6.utils.events import LOGGER, NCOLS, load_yaml
from yolov6.utils.checkpoint import load_checkpoint, load_state_dict
from yolov6.utils.config import Config
from yolov6.assigners.anchor_generator import generate_anchors
from yolov6.utils.general import dist2bbox

from tools_nndct.quantization.eval import EvalerWrapper
from torch_rewriters import MODULE_REWRITER, RewriterContext, patch_model
from torch_rewriters.utils import IR, Backend


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./configs/repopt/yolov6s_opt_qat.py', help='model config')
    parser.add_argument('--weights', type=str, default='./yolov6s.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--calib-batch-number', type=int, default=1000, help='calib batch number')
    parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
    parser.add_argument('--inplace', action='store_true', help='set Detect() inplace=True')
    parser.add_argument('--eval-float', action='store_true', help='eval float model')
    parser.add_argument('--qat', action='store_true', help='export qat model')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0, 1, 2, 3 or cpu')
    parser.add_argument('--data-yaml', type=str, default='./data/coco.yaml', help='data config')
    parser.add_argument('--eval-yaml', type=str, default='./tools_nndct/quantization/eval.yaml', help='evaluation config')
    args = parser.parse_args()
    args.img_size *= 2 if len(args.img_size) == 1 else 1  # expand
    print(args)
    eval_cfg = load_yaml(args.eval_yaml)
    eval_cfg['batch_size'] = args.batch_size
    eval_cfg['data'] = args.data_yaml
    eval_cfg['half'] = args.half
    eval_cfg['device'] = args.device
    yolov6_evaler = EvalerWrapper(eval_cfg=eval_cfg)
    val_loader = EvalerWrapper(eval_cfg=eval_cfg).val_loader
    eval_cfg_float = copy.deepcopy(eval_cfg)
    eval_cfg_float['batch_size'] = 32
    if args.eval_float:
        yolov6_evaler_float = EvalerWrapper(eval_cfg=eval_cfg_float)
    # Check device
    cuda = args.device != 'cpu' and torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    assert not (device.type == 'cpu' and args.half), '--half only compatible with GPU export, i.e. use --device 0'
    # Load PyTorch model

    cfg = Config.fromfile(args.conf)
    data_cfg = load_yaml(args.data_yaml)
    if not hasattr(cfg, 'conv_act'):
        setattr(cfg, 'conv_act', 'relu')
    if cfg.conv_act == 'leaky_relu':
        conv_act_builder = lambda: nn.LeakyReLU(negative_slope=26./256, inplace=True)
    else: 
        conv_act_builder = lambda: nn.ReLU(inplace=True)
    torch_rewriters_cfg = dict(conv_act_builder=conv_act_builder)

    with RewriterContext(cfg=torch_rewriters_cfg, backend=Backend.NNDCT.value, ir=IR.XMODEL), torch.no_grad():
        nc = yolov6_evaler.val.data['nc']
        names = yolov6_evaler.val.data['names']
        model = build_model(cfg=cfg, num_classes=nc, device=device)
        model = load_state_dict(args.weights, model, map_location=device)
        model.nc, model.names = nc, names
        LOGGER.info(f'Loading state_dict from {args.weights} for post quantization...')
        # model = load_checkpoint(args.weights, map_location=device, inplace=True, fuse=False)  # load FP32 model
        model.to(device)
        model.eval()

        for layer in model.modules():
            if isinstance(layer, RepVGGBlock):
                layer.switch_to_deploy()

        for k, m in model.named_modules():
            if isinstance(m, Conv):  # assign export-friendly activations
                if isinstance(m.act, nn.SiLU):
                    m.act = SiLU()
            elif isinstance(m, Detect):
                m.inplace = args.inplace

        nndct_out = Path(args.weights).parent
        original_model = model

        image = next(iter(val_loader))[0]
        image = image.float()/255.0
        image = image.to(device)
        # dry run
        y = original_model(image)
        from pytorch_nndct.utils.summary import model_complexity
        total_macs, total_params = model_complexity(model, image, readable=True, print_model_analysis=True)
        print(f"Macs: {total_macs} Params: {total_params}")
        if args.eval_float:
            quant_mAP = yolov6_evaler_float.eval(original_model)
            print("Float model mAP0.5={:.4f}, mAP0.5_0.95={:0.4f}".format(quant_mAP[0], quant_mAP[1]))

        if args.qat:
            # Step1: load qat trainable model weights
            model.train()
            from pytorch_nndct import QatProcessor
            ckpt = torch.load(args.weights, map_location=device)
            if 'nndct_qat_trainable_ema_model' in ckpt:
                state_dict = ckpt['nndct_qat_trainable_ema_model']
                # workaround
                # since ema model has not trained, which will cause nndct raise a exception that QAT model must be trained to convert to deployable model
                for k, v in state_dict.items():
                    if 'warmup_enabled' in k:
                        state_dict[k] = torch.zeros_like(state_dict[k])
            else:
                state_dict = ckpt['nndct_qat_trainable_model']

            qat_processor = QatProcessor(
                model, image, bitwidth=8, device=device)
            torch_rewriters_cfg['qat_processor'] = qat_processor
            torch_rewriters_cfg['model'] = model
            quantized_model = qat_processor.trainable_model()
            quantized_model.load_state_dict(state_dict, strict=True)
            deployable_model = qat_processor.convert_to_deployable(quantized_model, output_dir=nndct_out.as_posix())
            model.load_state_dict(deployable_model.state_dict(), strict=True)
            model.eval()
            del ckpt, quantized_model, deployable_model
            print("Load QAT model finished.")
        else:
            # Step1: do post training calibration
            from pytorch_nndct.apis import torch_quantizer
            quantizer = torch_quantizer('calib', original_model, input_args=(image,), output_dir=nndct_out.as_posix(), bitwidth=8)
            model_ptq = quantizer.quant_model
            model_ptq_wrapper = ModelNNDctWrapper(original_model, model_ptq)

            pbar = tqdm(yolov6_evaler.val_loader, desc="Calibrate model in val datasets.", ncols=NCOLS, total=args.calib_batch_number)
            for i, (imgs, targets, paths, shapes) in enumerate(pbar):
                imgs = imgs.to(device, non_blocking=True)
                imgs = imgs.half() if args.half else imgs.float()
                imgs /= 255
                # Calib
                outputs, _ = model_ptq_wrapper(imgs)
                if i >= args.calib_batch_number:
                    break

            quantizer.export_quant_config()
            print("Post Training Quantization Calibration finished.")

        # Step2: do quant dump & test
        # dry run
        y = original_model(image)

        from pytorch_nndct.apis import torch_quantizer
        quantizer = torch_quantizer('test', original_model, input_args=(image,), output_dir=nndct_out.as_posix(), bitwidth=8)
        model_ptq = quantizer.quant_model
        model_ptq_wrapper = ModelNNDctWrapper(original_model, model_ptq)
        model_ptq_wrapper(image)
        quantizer.export_xmodel(output_dir=nndct_out.as_posix(), deploy_check=True)
        quantizer.export_onnx_model(output_dir=nndct_out.as_posix(), verbose=True, dynamic_batch=True)
        quant_mAP = yolov6_evaler.eval(model_ptq_wrapper)
        print("Post Training Quantization model mAP0.5={:.4f}, mAP0.5_0.95={:0.4f}".format(quant_mAP[0], quant_mAP[1]))

