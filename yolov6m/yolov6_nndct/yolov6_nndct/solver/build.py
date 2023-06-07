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


@FUNCTION_REWRITER.register_rewriter(func_name='yolov6.solver.build.build_optimizer', backend=Backend.NNDCT.value, ir=IR.XMODEL)
def build_optimizer_nndct(ctx, cfg, model):
    """ Build optimizer from cfg file."""
    g_bnw, g_w, g_b = [], [], []
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            g_b.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):
            g_bnw.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            g_w.append(v.weight)

    assert cfg.solver.optim in ['SGD', 'Adam', 'AdamW'], 'ERROR: unknown optimizer, use SGD defaulted'
    if cfg.solver.optim == 'SGD':
        optimizer = torch.optim.SGD(g_bnw, lr=cfg.solver.lr0, momentum=cfg.solver.momentum, nesterov=True)
    elif cfg.solver.optim == 'Adam':
        optimizer = torch.optim.Adam(g_bnw, lr=cfg.solver.lr0, betas=(cfg.solver.momentum, 0.999))
    elif cfg.solver.optim == 'AdamW':
        optimizer = torch.optim.AdamW(g_bnw, lr=cfg.solver.lr0, betas=(cfg.solver.momentum, 0.999))

    optimizer.add_param_group({'params': g_w, 'weight_decay': cfg.solver.weight_decay})
    optimizer.add_param_group({'params': g_b})

    threshold = [
        param for name, param in model.named_parameters()
        if 'threshold' in name
    ]
    g_qat = {
        'params': threshold,
        'lr': cfg.solver.lr0 * 100,
        'name': 'threshold'
    }
    optimizer.add_param_group(g_qat)
    LOGGER.info(f"{'optimizer:'} {type(optimizer).__name__} with parameter groups "
                f"{len(g_bnw)} weight, {len(g_w)} weight (no decay), {len(g_b)} bias, {len(g_qat)} log_threshold")

    del g_bnw, g_w, g_b, g_qat
    return optimizer