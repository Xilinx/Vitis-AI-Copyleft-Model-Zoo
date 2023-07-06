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
import torch
from yolov6.utils.events import LOGGER
from torch_rewriters import FUNCTION_REWRITER
from torch_rewriters.utils import IR, Backend


def _clean_k(k):
    k = k.replace('._module.', '.').replace('_module.', '')
    return k


@FUNCTION_REWRITER.register_rewriter(func_name='yolov6.utils.checkpoint.load_state_dict', backend=Backend.NNDCT.value, ir=IR.XMODEL)
def load_state_dict_nndct(ctx, weights, model, map_location=None, state_dict=None):
    """Load weights from checkpoint file, only assign weights those layers' name and shape are match."""
    ckpt = torch.load(weights, map_location=map_location)
    if state_dict is None:
        if ckpt.get('ema', None) is not None:
            state_dict = ckpt['ema'].float().state_dict()
        else:
            state_dict = ckpt['model'].float().state_dict()
    model_state_dict = model.state_dict()
    state_dict = {_clean_k(k): v for k, v in state_dict.items()}
    model_state_dict_keys_map = {_clean_k(k): k for k, v in model_state_dict.items()}
    model_state_dict = {_clean_k(k): v for k, v in model_state_dict.items()}
    new_state_dict = {}
    for k in model_state_dict.keys():
        if k in state_dict and model_state_dict[k].shape == state_dict[k].shape:
            new_state_dict[model_state_dict_keys_map[k]] = state_dict[k]
        else:
            if k not in state_dict:
                LOGGER.warning(f'Missing params with key: {model_state_dict_keys_map[k]} in weights')
            else:
                LOGGER.warning(f'Mismatch params\' shape with key: {model_state_dict_keys_map[k]} in weights, expected: {model_state_dict[k].shape} but found: {state_dict[k].shape}')

    model.load_state_dict(new_state_dict, strict=False)
    del ckpt, state_dict, model_state_dict
    return model


@FUNCTION_REWRITER.register_rewriter(func_name='yolov6.utils.checkpoint.strip_optimizer', backend=Backend.NNDCT.value, ir=IR.XMODEL)
def strip_optimizer_nndct(ctx, ckpt_dir, epoch):
    # Do not strip optimizer
    pass
