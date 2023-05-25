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
import os
import os.path as osp
import sys
import torch

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from yolov6.core.evaler import Evaler
from yolov6.utils.events import LOGGER
from yolov6.utils.general import increment_name


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='YOLOv6 PyTorch Evalating', add_help=add_help)
    parser.add_argument('--data', type=str, default='./data/coco.yaml', help='dataset yaml file path.')
    parser.add_argument('--weights', type=str, default='./yolov6s.engine', help='tensorrt engine file path.')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--task', default='val', help='can only be val now.')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save_dir', type=str, default='runs/val/', help='evaluation save dir')
    parser.add_argument('--name', type=str, default='exp', help='save evaluation results to save_dir/name')
    args = parser.parse_args()
    LOGGER.info(args)
    return args


@torch.no_grad()
def run(data,
        weights=None,
        batch_size=32,
        img_size=640,
        task='val',
        device='',
        save_dir='',
        name = ''
        ):
    """ 
    TensorRT models's evaluation process.
    """

     # task
    assert task== 'val', f'task type can only be val, however you set it to {task}'

    save_dir = str(increment_name(osp.join(save_dir, name)))
    os.makedirs(save_dir, exist_ok=True)

    dummy_model = torch.zeros(0)
    device = Evaler.reload_device(device, dummy_model, task)

    data = Evaler.reload_dataset(data) if isinstance(data, str) else data

    # init
    val = Evaler(data, batch_size, img_size, None, \
                None, device, False, save_dir)

    dataloader,pred_result = val.eval_trt(weights)
    eval_result = val.eval_model(pred_result, dummy_model, dataloader, task)
    return eval_result


def main(args):
    run(**vars(args))


if __name__ == "__main__":
    args = get_args_parser()
    main(args)
