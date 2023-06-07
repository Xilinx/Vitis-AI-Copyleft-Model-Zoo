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

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_rewriters import MODULE_REWRITER
from torch_rewriters.utils import IR, Backend


@MODULE_REWRITER.register_rewrite_module('yolov6.models.reppan.RepPANNeck', backend=Backend.NNDCT.value, ir=IR.XMODEL)
class RepPANNeckNNDct(nn.Module):
    def __init__(self, module, cfg):
        super().__init__()
        from pytorch_nndct.nn.modules import functional as nF
        module.cat = nF.Cat()
        module.cat2 = nF.Cat()
        module.cat3 = nF.Cat()
        module.cat4 = nF.Cat()
        self._module = module

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._module, name) 

    def forward(self, input):
        true_self = self
        self = true_self._module
        (x2, x1, x0) = input

        fpn_out0 = self.reduce_layer0(x0)
        upsample_feat0 = self.upsample0(fpn_out0)
        f_concat_layer0 = self.cat([upsample_feat0, x1], 1)
        f_out0 = self.Rep_p4(f_concat_layer0)

        fpn_out1 = self.reduce_layer1(f_out0)
        upsample_feat1 = self.upsample1(fpn_out1)
        f_concat_layer1 = self.cat2([upsample_feat1, x2], 1)
        pan_out2 = self.Rep_p3(f_concat_layer1)

        down_feat1 = self.downsample2(pan_out2)
        p_concat_layer1 = self.cat3([down_feat1, fpn_out1], 1)
        pan_out1 = self.Rep_n3(p_concat_layer1)

        down_feat0 = self.downsample1(pan_out1)
        p_concat_layer2 = self.cat4([down_feat0, fpn_out0], 1)
        pan_out0 = self.Rep_n4(p_concat_layer2)

        outputs = [pan_out2, pan_out1, pan_out0]

        return outputs


@MODULE_REWRITER.register_rewrite_module('yolov6.models.reppan.CSPRepPANNeck', backend=Backend.NNDCT.value, ir=IR.XMODEL)
class CSPRepPANNeckNNDct(nn.Module):
    def __init__(self, module, cfg):
        super().__init__()
        from pytorch_nndct.nn.modules import functional as nF
        module.cat = nF.Cat()
        module.cat2 = nF.Cat()
        module.cat3 = nF.Cat()
        module.cat4 = nF.Cat()
        self._module = module

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._module, name) 

    def forward(self, input):
        true_self = self
        self = true_self._module
        (x2, x1, x0) = input

        fpn_out0 = self.reduce_layer0(x0)
        upsample_feat0 = self.upsample0(fpn_out0)
        f_concat_layer0 = self.cat([upsample_feat0, x1], 1)
        f_out0 = self.Rep_p4(f_concat_layer0)

        fpn_out1 = self.reduce_layer1(f_out0)
        upsample_feat1 = self.upsample1(fpn_out1)
        f_concat_layer1 = self.cat2([upsample_feat1, x2], 1)
        pan_out2 = self.Rep_p3(f_concat_layer1)

        down_feat1 = self.downsample2(pan_out2)
        p_concat_layer1 = self.cat3([down_feat1, fpn_out1], 1)
        pan_out1 = self.Rep_n3(p_concat_layer1)

        down_feat0 = self.downsample1(pan_out1)
        p_concat_layer2 = self.cat4([down_feat0, fpn_out0], 1)
        pan_out0 = self.Rep_n4(p_concat_layer2)

        outputs = [pan_out2, pan_out1, pan_out0]

        return outputs
