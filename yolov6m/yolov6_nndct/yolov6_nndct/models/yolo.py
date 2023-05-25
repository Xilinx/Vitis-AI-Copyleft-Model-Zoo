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
from torch import nn
from yolov6.models.yolo import Model
from torch_rewriters import MODULE_REWRITER, FUNCTION_REWRITER, patch_model
from torch_rewriters.utils import IR, Backend

@FUNCTION_REWRITER.register_rewriter(func_name='yolov6.models.yolo.build_model', backend=Backend.NNDCT.value, ir=IR.XMODEL)
def build_model_nndct(ctx, cfg, num_classes, device):
    model = Model(cfg, channels=3, num_classes=num_classes, anchors=cfg.model.head.anchors).to(device)
    model = patch_model(model, cfg=ctx.cfg, backend=Backend.NNDCT.value, ir=IR.XMODEL)
    return model


@MODULE_REWRITER.register_rewrite_module('yolov6.models.yolo.Model', backend=Backend.NNDCT.value, ir=IR.XMODEL)
class ModelNNDct(nn.Module):
    def __init__(self, module, cfg):
        super().__init__()
        self._module = module
        from pytorch_nndct.nn import QuantStub, DeQuantStub
        self.quant = QuantStub()
        if hasattr(module, 'nc'):
            self.nc = module.nc
        if hasattr(module, 'names'):
            self.names = module.names

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._module, name) 

    def forward(self, x):
        true_self = self
        self = true_self._module
        
        x = true_self.quant(x)
        export_mode = torch.onnx.is_in_onnx_export()
        x = self.backbone(x)
        x = self.neck(x)
        if export_mode == False:
            featmaps = []
            featmaps.extend(x)
        x = self.detect(x)
        return x if export_mode is True else [x, featmaps]


class ModelNNDctWrapper(nn.Module):
    def __init__(self, module, module_quant):
        super().__init__()
        self._module = module
        self._module_quant = module_quant

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            try:
                return getattr(self._module_quant, name) 
            except AttributeError:
                if self._module is not None:
                    return getattr(self._module, name) 
                else:
                    raise


    def forward(self, x):
        outs = self._module_quant(x)
        return outs[0], outs[1:]