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

import os
import math

import torch
import torch.nn as nn

from yolov6.utils.events import LOGGER
from torch_rewriters import FUNCTION_REWRITER
from torch_rewriters.utils import IR, Backend


@FUNCTION_REWRITER.register_rewriter(func_name='yolov6.utils.RepOptimizer.get_optimizer_param', backend=Backend.NNDCT.value, ir=IR.XMODEL)
def get_optimizer_param_nndct(ctx, args, cfg, model):
    """ Build optimizer from cfg file."""
    accumulate = max(1, round(64 / args.batch_size))
    cfg.solver.weight_decay *= args.batch_size * accumulate / 64

    g_bnw, g_w, g_b = [], [], []
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            g_b.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):
            g_bnw.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            g_w.append(v.weight)
    threshold = [
        param for name, param in model.named_parameters()
        if 'threshold' in name
    ]
    g_qat = {
        'params': threshold,
        'lr': cfg.solver.lr0 * 100,
        'name': 'threshold'
    }
    return [{'params': g_bnw},
            {'params': g_w, 'weight_decay': cfg.solver.weight_decay},
            {'params': g_b}, g_qat]

