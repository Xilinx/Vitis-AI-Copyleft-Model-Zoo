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

# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Experimental modules
"""
from copy import deepcopy
from functools import partial
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from pytorch_nndct.nn.modules import functional as nF

from models.common import Conv, TransformerLayer, TransformerBlock, GhostBottleneck, Bottleneck, BottleneckCSP, C3, SPP, Focus, GhostConv, Concat, Classify, SPPF
from utils.general import check_img_size
from utils.downloads import attempt_download


class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Sum(nn.Module):
    # Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, n, weight=False):  # n: number of inputs
        super().__init__()
        self.weight = weight  # apply weights boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            self.w = nn.Parameter(-torch.arange(1., n) / 2, requires_grad=True)  # layer weights

    def forward(self, x):
        y = x[0]  # no weight
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


class MixConv2d(nn.Module):
    # Mixed Depth-wise Conv https://arxiv.org/abs/1907.09595
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):
        super().__init__()
        groups = len(k)
        if equal_ch:  # equal c_ per group
            i = torch.linspace(0, groups - 1E-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(groups)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [c2] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList([nn.Conv2d(c1, int(c_[g]), k[g], s, k[g] // 2, bias=False) for g in range(groups)])
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return x + self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = []
        for module in self:
            y.append(module(x, augment, profile, visualize)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


def attempt_load(weights, map_location=None, inplace=True, fuse=True, force_reexport_deployable_model=False, imgsz=640):
    from models.yolo import Detect, Model

    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(attempt_download(w), map_location=map_location)  # load
        _model_ckpt = ckpt['ema' if ckpt.get('ema') else 'model']
        _model = Model(ckpt['model'].yaml, ch=3, nc=_model_ckpt.yaml['nc'], anchors=_model_ckpt.hyp.get('anchors'))
        _model.load_state_dict(_model_ckpt.state_dict(), strict=True)
        _model.nc = _model_ckpt.nc  # attach number of classes to model
        _model.hyp = _model_ckpt.hyp  # attach hyperparameters to model
        _model.class_weights = _model_ckpt.class_weights  # attach class weights
        _model.names = _model_ckpt.names
        _model.to(next(_model_ckpt.parameters()).device)

        # This is the case we run validation using QAT deployable model
        if 'qat_model_quant_info' in ckpt:
            if force_reexport_deployable_model:
                from pytorch_nndct import QatProcessor
                # Image sizes
                _ori_model = deepcopy(_model)
                _model.train()
                gs = max(int(_model.stride.max()), 32)  # grid size (max stride)
                imgsz = check_img_size(imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple
                im = torch.zeros(1, 3, imgsz, imgsz).to(next(_model_ckpt.parameters()).device)  # image size(1,3,320,192) BCHW iDetection
                # dry run
                for _ in range(2):
                    y = _model(im)  # dry runs
                _model.forward = partial(_model.forward, augment=False, profile=False, visualize=False, quant=True)
                qat_processor = QatProcessor(_model, (im,), bitwidth=8, mix_bit=False)
                calib_dir = Path(w).parent / 'nndct_quant'
                _trainable_model = qat_processor.trainable_model()
                _trainable_model.load_state_dict(ckpt['qat_ema_state_dict'], strict=True)
                _deployable_net = qat_processor.convert_to_deployable(_trainable_model, calib_dir.as_posix())
                _ori_model.load_state_dict(_deployable_net.state_dict(), strict=True)
                _model = _ori_model
            else:
                def write_quant_info(dir, quant_info_str):
                    dir.mkdir(exist_ok=True, parents=True)
                    with open(dir / 'quant_info.json', 'w') as f:
                        f.write(quant_info_str)
                w = Path(w)
                w_dir = w.parent
                write_quant_info(w_dir / 'nndct_quant', ckpt['qat_model_quant_info'])
        if fuse:
            model.append(_model.float().fuse().eval())  # FP32 model
        else:
            model.append(_model.float().eval())  # without layer fuse


    # # Compatibility updates
    # for m in model.modules():
    #     if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model]:
    #         m.inplace = inplace  # pytorch 1.7.0 compatibility
    #         if type(m) is Detect:
    #             if not isinstance(m.anchor_grid, list):  # new Detect Layer compatibility
    #                 delattr(m, 'anchor_grid')
    #                 setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
    #             if nndct_utils.is_running_quant():
    #                 from pytorch_nndct.nn import QuantStub, DeQuantStub
    #                 m.add_module('dequant', DeQuantStub())
    #             else:
    #                 m.add_module('dequant', nn.Identity())
    #         if type(m) is Model:
    #             if nndct_utils.is_running_quant():
    #                 from pytorch_nndct.nn import QuantStub, DeQuantStub
    #                 m.add_module('quant', QuantStub())
    #             else:
    #                 m.add_module('quant', nn.Identity())
    #     elif type(m) is Conv:
    #         m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
    #     elif type(m) is TransformerLayer:
    #         if not hasattr(m, 'add1'):
    #             m.add_module('add1', nF.Add())
    #             m.add_module('add2', nF.Add())
    #     elif type(m) in (TransformerBlock, GhostBottleneck):
    #         if not hasattr(m, 'add'):
    #             m.add_module('add', nF.Add())
    #     elif type(m) is Bottleneck:
    #         if not hasattr(m, 'skip_add'):
    #             m.add_module('skip_add', nF.Add())
    #     elif type(m) in (BottleneckCSP, C3, SPP, Focus, GhostConv, Concat, Classify):
    #         m.add_module('cat', nF.Cat())
    #     elif type(m) is SPPF:
    #         if not hasattr(m, 'm1'):
    #             m.add_module('cat', nF.Cat())
    #             m.add_module('m1', nn.MaxPool2d(kernel_size=m.m.kernel_size, stride=m.m.stride, padding=m.m.padding))
    #             m.add_module('m2', nn.MaxPool2d(kernel_size=m.m.kernel_size, stride=m.m.stride, padding=m.m.padding))
    #             m.add_module('m3', nn.MaxPool2d(kernel_size=m.m.kernel_size, stride=m.m.stride, padding=m.m.padding))

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print(f'Ensemble created with {weights}\n')
        for k in ['names']:
            setattr(model, k, getattr(model[-1], k))
        model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
        return model  # return ensemble


def attempt_load_qat(weights, map_location=None, imgsz=640):
    from models.yolo import Detect, Model

    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(attempt_download(w), map_location=map_location)  # load
        _model_ckpt = ckpt['ema' if ckpt.get('ema') else 'model']
        _model = Model(ckpt['model'].yaml, ch=3, nc=_model_ckpt.yaml['nc'], anchors=_model_ckpt.hyp.get('anchors'))
        # _model.load_state_dict(_model_ckpt.state_dict(), strict=True)
        _model.nc = _model_ckpt.nc  # attach number of classes to model
        _model.hyp = _model_ckpt.hyp  # attach hyperparameters to model
        _model.class_weights = _model_ckpt.class_weights  # attach class weights
        _model.names = _model_ckpt.names
        _model.to(next(_model_ckpt.parameters()).device)
        
        from pytorch_nndct import QatProcessor
        # Image sizes
        _model.train()
        gs = max(int(_model.stride.max()), 32)  # grid size (max stride)
        nl = _model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
        imgsz = check_img_size(imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple
        im = torch.zeros(1, 3, imgsz, imgsz).to(next(_model_ckpt.parameters()).device)  # image size(1,3,320,192) BCHW iDetection
        # dry run
        for _ in range(2):
            y = _model(im)  # dry runs
        _model.forward = partial(_model.forward, augment=False, profile=False, visualize=False, quant=True)
        qat_processor = QatProcessor(_model, (im,), bitwidth=8, mix_bit=False)
        calib_dir = Path(w).parent / 'nndct_quant'
        _trainable_model = qat_processor.trainable_model(calib_dir=calib_dir.as_posix())
        _model.load_state_dict(_trainable_model.state_dict(), strict=True)

        model.append(_model.float().eval())  # without layer fuse

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print(f'Ensemble created with {weights}\n')
        for k in ['names']:
            setattr(model, k, getattr(model[-1], k))
        model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
        return model  # return ensemble
