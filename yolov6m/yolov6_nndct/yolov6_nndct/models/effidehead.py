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

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from yolov6.layers.common import *
from yolov6.assigners.anchor_generator import generate_anchors
from yolov6.utils.general import dist2bbox
from torch_rewriters import MODULE_REWRITER
from torch_rewriters.utils import IR, Backend


@MODULE_REWRITER.register_rewrite_module('yolov6.models.effidehead.Detect', backend=Backend.NNDCT.value, ir=IR.XMODEL)
class Detect(nn.Module):
    def __init__(self, module, cfg):
        super().__init__()
        from pytorch_nndct.nn import DeQuantStub
        self.is_nndct_qat = cfg.get('is_nndct_qat', False)
        self.dequant_lst = nn.ModuleList([DeQuantStub() for i in range(module.nl * 2)])
        module.softmax_lst = nn.ModuleList([nn.Softmax(dim=1) for i in range(module.nl)])
        module.proj_conv_lst = nn.ModuleList([copy.deepcopy(module.proj_conv) for i in range(module.nl)])
        self._module = module

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._module, name) 

    def initialize_biases(self):
        true_self = self
        self = true_self._module

        for conv in self.cls_preds:
            b = conv.bias.view(-1, )
            b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)

        for conv in self.reg_preds:
            b = conv.bias.view(-1, )
            b.data.fill_(1.0)
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)
        
        self.proj = nn.Parameter(torch.linspace(0, self.reg_max, self.reg_max + 1), requires_grad=False)
        self.proj_conv.weight = nn.Parameter(self.proj.view([1, self.reg_max + 1, 1, 1]).clone().detach(),
                                                   requires_grad=False)
        for proj_conv in self.proj_conv_lst:
            proj_conv.weight = nn.Parameter(self.proj.view([1, self.reg_max + 1, 1, 1]).clone().detach(),
                                            requires_grad=False)

    def forward(self, x):
        true_self = self
        self = true_self._module
        x_numpy = [a.detach().cpu().numpy() for a in x]
        if self.training:
            cls_score_list = []
            reg_bboxes_list = []
            reg_distri_list = []

            for i in range(self.nl):
                b, _, h, w = x_numpy[i].shape
                l = h * w

                x[i] = self.stems[i](x[i])
                cls_x = x[i]
                reg_x = x[i]
                cls_feat = self.cls_convs[i](cls_x)
                cls_output = self.cls_preds[i](cls_feat)
                reg_feat = self.reg_convs[i](reg_x)
                reg_output = self.reg_preds[i](reg_feat)
                if self.use_dfl and true_self.is_nndct_qat:
                    reg_output_bboxes = reg_output.reshape([-1, 4, self.reg_max + 1, l]).permute(0, 2, 1, 3)
                    reg_output_bboxes = self.proj_conv_lst[i](self.softmax_lst[i](reg_output_bboxes)).reshape(b, 4, 1, l)
                    reg_bboxes_list.append(reg_output_bboxes.flatten(2).permute((0, 2, 1)))

                cls_output = torch.sigmoid(cls_output)
                cls_score_list.append(cls_output.flatten(2).permute((0, 2, 1)))
                reg_distri_list.append(reg_output.flatten(2).permute((0, 2, 1)))
            
            cls_score_list = torch.cat(cls_score_list, axis=1)
            reg_distri_list = torch.cat(reg_distri_list, axis=1)
            if self.use_dfl and true_self.is_nndct_qat:
                reg_bboxes_list = torch.cat(reg_bboxes_list, axis=1)
                return x, cls_score_list, reg_bboxes_list, reg_distri_list

            return x, cls_score_list, reg_distri_list
        else:
            cls_score_list = []
            reg_dist_list = []
            # The shape OP will be traced by NNDCT, we need to detach it from xmodel
            x_numpy = [a.detach().cpu().numpy() for a in x]
            anchor_points, stride_tensor = generate_anchors(
                x_numpy, self.stride, self.grid_cell_size, self.grid_cell_offset, device=x[0].device, is_eval=True)

            for i in range(self.nl):
                b, _, h, w = x_numpy[i].shape
                l = h * w
                x[i] = self.stems[i](x[i])
                cls_x = x[i]
                reg_x = x[i]
                cls_feat = self.cls_convs[i](cls_x)
                cls_output = self.cls_preds[i](cls_feat)
                reg_feat = self.reg_convs[i](reg_x)
                reg_output = self.reg_preds[i](reg_feat)


                if self.use_dfl:
                    reg_output = reg_output.reshape([-1, 4, self.reg_max + 1, l]).permute(0, 2, 1, 3)
                    reg_output = self.proj_conv_lst[i](self.softmax_lst[i](reg_output))
                cls_output = torch.sigmoid(cls_output)
                # Dequant
                reg_output = true_self.dequant_lst[i*2+1](reg_output)
                cls_output = true_self.dequant_lst[i*2](cls_output)
                cls_score_list.append(cls_output.reshape([b, self.nc, l]))
                reg_dist_list.append(reg_output.reshape([b, 4, l]))
            
            cls_score_list = torch.cat(cls_score_list, axis=-1).permute(0, 2, 1)
            reg_dist_list = torch.cat(reg_dist_list, axis=-1).permute(0, 2, 1)

            pred_bboxes = dist2bbox(reg_dist_list, anchor_points, box_format='xywh')
            pred_bboxes *= stride_tensor
            return torch.cat(
                [
                    pred_bboxes,
                    torch.ones((b, pred_bboxes.shape[1], 1), device=pred_bboxes.device, dtype=pred_bboxes.dtype),
                    cls_score_list
                ],
                axis=-1)
