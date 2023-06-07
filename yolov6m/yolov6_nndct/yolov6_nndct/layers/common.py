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

import warnings
import types
import copy

import numpy as np
import torch
import torch.nn as nn
from yolov6.layers.common import RepVGGBlock, RealVGGBlock, LinearAddBlock
from torch_rewriters import MODULE_REWRITER
from torch_rewriters.utils import IR, Backend


@MODULE_REWRITER.register_rewrite_module('yolov6.layers.common.Conv', backend=Backend.NNDCT.value, ir=IR.XMODEL)
class ConvNNDct(nn.Module):
    '''Normal Conv with ReLU activation'''
    def __init__(self, module, cfg):
        super().__init__()
        act_builder = cfg.get('conv_act_builder', lambda: nn.ReLU(inplace=True))
        module.act = act_builder()
        self._module = module

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._module, name) 
    
    def forward(self, x):
        return self._module(x)

    def forward_fuse(self, x):
        return self._module.act(self._module.conv(x))
    
# TODO remove
# For compatbility
Conv = ConvNNDct

@MODULE_REWRITER.register_rewrite_module('yolov6.layers.common.BottleRep', backend=Backend.NNDCT.value, ir=IR.XMODEL)
class BottleRepNNDct(nn.Module):
    def __init__(self, module, cfg):
        super().__init__()
        from pytorch_nndct.nn.modules import functional as nF
        module.add = nF.Add()
        self._module = module
        if isinstance(module.conv2, (RealVGGBlock, LinearAddBlock)):
            c = module.conv2.bn.num_features
        else:
            raise NotImplementedError
        self.alpha_conv_c = c
        self.alpha_conv = nn.Conv2d(c, c, 1, 1, 0, groups=c, bias=False)

        def _load_from_state_dict(obj, state_dict, prefix, local_metadata, strict,
                                missing_keys, unexpected_keys, error_msgs):
            self._update_alpha()
            super(type(obj), obj)._load_from_state_dict(state_dict, prefix, local_metadata, False,
                                missing_keys, unexpected_keys, error_msgs)
        f = types.MethodType(_load_from_state_dict, self.alpha_conv)
        self.alpha_conv._load_from_state_dict = f
        self._update_alpha()

    def _update_alpha(self):
        alpha = self._module.alpha.data.detach().cpu().numpy()
        weights = np.zeros((self.alpha_conv_c, 1, 1, 1), dtype=alpha.dtype)
        weights[...] = alpha
        self.alpha = alpha
        self.alpha_conv.weight = nn.Parameter(torch.from_numpy(weights).to(next(self._module.parameters()).device), requires_grad=False)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._module, name) 

    def eval(self):
        self._update_alpha() 
        return super().eval()

    def forward(self, x):
        true_self = self
        self = true_self._module
        if self.training:
            outputs = self.conv1(x)
            outputs = self.conv2(outputs)
            return self.add(outputs, self.alpha * x) if self.shortcut else outputs
        else:
            outputs = self.conv1(x)
            outputs = self.conv2(outputs)
            if not (self.alpha == 1).all():
                return self.add(outputs, true_self.alpha_conv(x)) if self.shortcut else outputs
            else:
                return self.add(outputs, x) if self.shortcut else outputs
  

@MODULE_REWRITER.register_rewrite_module('yolov6.layers.common.SimSPPF', backend=Backend.NNDCT.value, ir=IR.XMODEL)
class SimSPPFNNDct(nn.Module):
    '''Simplified SPPF with ReLU activation'''
    def __init__(self, module, cfg):
        super().__init__()
        from pytorch_nndct.nn.modules import functional as nF
        module.cat = nF.Cat()
        module.m2 = copy.deepcopy(module.m)
        module.m3 = copy.deepcopy(module.m)
        self._module = module

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._module, name) 

    def forward(self, x):
        true_self = self
        self = true_self._module
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.m(x)
            y2 = self.m2(y1)
            return self.cv2(self.cat([x, y1, y2, self.m3(y2)], 1))
            

@MODULE_REWRITER.register_rewrite_module('yolov6.layers.common.BepC3', backend=Backend.NNDCT.value, ir=IR.XMODEL)
class BepC3NNDct(nn.Module):
    def __init__(self, module, cfg):
        super().__init__()
        from pytorch_nndct.nn.modules import functional as nF
        module.cat = nF.Cat()
        self._module = module

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._module, name) 

    def forward(self, x):
        true_self = self
        self = true_self._module
        if self.concat is True:
            return self.cv3(self.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
        else:
            return self.cv3(self.m(self.cv1(x)))
            
