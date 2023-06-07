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
import numpy as np
import torch.nn.functional as F
from yolov6.assigners.anchor_generator import generate_anchors
from yolov6.utils.general import dist2bbox, bbox2dist, xywh2xyxy
from yolov6.utils.figure_iou import IOUloss
from yolov6.assigners.atss_assigner import ATSSAssigner
from yolov6.assigners.tal_assigner import TaskAlignedAssigner
from torch_rewriters import MODULE_REWRITER, FUNCTION_REWRITER, patch_model
from torch_rewriters.utils import IR, Backend


@FUNCTION_REWRITER.register_rewriter(func_name='yolov6.models.loss_distill.ComputeLoss.__call__', backend=Backend.NNDCT.value, ir=IR.XMODEL)
def ComputeLoss_distill____call__(
    ctx, 
    self,
    outputs,
    t_outputs,
    s_featmaps,
    t_featmaps,
    targets,
    epoch_num,
    max_epoch,
    temperature):
    is_nndct_qat = ctx.cfg.get('is_nndct_qat', False)
        
    if is_nndct_qat and self.use_dfl:
        feats, pred_scores, pred_bboxes, pred_distri = outputs
        t_feats, t_pred_scores, t_pred_bboxes, t_pred_distri = t_outputs
    else:
        feats, pred_scores, pred_distri = outputs
        t_feats, t_pred_scores, t_pred_distri = t_outputs
    anchors, anchor_points, n_anchors_list, stride_tensor = \
            generate_anchors(feats, self.fpn_strides, self.grid_cell_size, self.grid_cell_offset, device=feats[0].device)
    t_anchors, t_anchor_points, t_n_anchors_list, t_stride_tensor = \
            generate_anchors(t_feats, self.fpn_strides, self.grid_cell_size, self.grid_cell_offset, device=feats[0].device)

    assert pred_scores.type() == pred_distri.type()
    gt_bboxes_scale = torch.full((1,4), self.ori_img_size).type_as(pred_scores)
    batch_size = pred_scores.shape[0]

    # targets
    targets =self.preprocess(targets, batch_size, gt_bboxes_scale)
    gt_labels = targets[:, :, :1]
    gt_bboxes = targets[:, :, 1:] #xyxy
    mask_gt = (gt_bboxes.sum(-1, keepdim=True) > 0).float()
    
    # pboxes
    anchor_points_s = anchor_points / stride_tensor
    t_anchor_points_s = t_anchor_points / t_stride_tensor
    if is_nndct_qat and self.use_dfl:
        pred_bboxes = dist2bbox(pred_bboxes, anchor_points_s)
        t_pred_bboxes = dist2bbox(t_pred_bboxes, t_anchor_points_s)
    else:
        pred_bboxes = self.bbox_decode(anchor_points_s, pred_distri) #xyxy
        t_pred_bboxes = self.bbox_decode(t_anchor_points_s, t_pred_distri) #xyxy

    if epoch_num < self.warmup_epoch:
        target_labels, target_bboxes, target_scores, fg_mask = \
            self.warmup_assigner(
                anchors,
                n_anchors_list,
                gt_labels,
                gt_bboxes,
                mask_gt,
                pred_bboxes.detach() * stride_tensor)
    else:
        target_labels, target_bboxes, target_scores, fg_mask = \
            self.formal_assigner(
                pred_scores.detach(),
                pred_bboxes.detach() * stride_tensor,
                anchor_points,
                gt_labels,
                gt_bboxes,
                mask_gt)

    # rescale bbox
    target_bboxes /= stride_tensor

    # cls loss
    target_labels = torch.where(fg_mask > 0, target_labels, torch.full_like(target_labels, self.num_classes))
    one_hot_label = F.one_hot(target_labels, self.num_classes + 1)[..., :-1]
    loss_cls = self.varifocal_loss(pred_scores, target_scores, one_hot_label)

    target_scores_sum = target_scores.sum()
    loss_cls /= target_scores_sum
    
    # bbox loss
    loss_iou, loss_dfl, d_loss_dfl = self.bbox_loss(pred_distri, pred_bboxes, t_pred_distri, t_pred_bboxes, temperature, anchor_points_s,
                                                    target_bboxes, target_scores, target_scores_sum, fg_mask)
    
    logits_student = pred_scores
    logits_teacher = t_pred_scores
    distill_num_classes = self.num_classes
    d_loss_cls = self.distill_loss_cls(logits_student, logits_teacher, distill_num_classes, temperature)
    if self.distill_feat:
        d_loss_cw = self.distill_loss_cw(s_featmaps, t_featmaps)
    else:
        d_loss_cw = torch.tensor(0.).to(feats[0].device)
    import math
    distill_weightdecay = ((1 - math.cos(epoch_num * math.pi / max_epoch)) / 2) * (0.01- 1) + 1
    d_loss_dfl *= distill_weightdecay
    d_loss_cls *= distill_weightdecay
    d_loss_cw *= distill_weightdecay
    loss_cls_all = loss_cls + d_loss_cls * self.distill_weight['class']
    loss_dfl_all = loss_dfl + d_loss_dfl * self.distill_weight['dfl']
    loss = self.loss_weight['class'] * loss_cls_all + \
            self.loss_weight['iou'] * loss_iou + \
            self.loss_weight['dfl'] * loss_dfl_all + \
            self.loss_weight['cwd'] * d_loss_cw
    
    return loss, \
        torch.cat(((self.loss_weight['iou'] * loss_iou).unsqueeze(0), 
                        (self.loss_weight['dfl'] * loss_dfl_all).unsqueeze(0),
                        (self.loss_weight['class'] * loss_cls_all).unsqueeze(0),
                        (self.loss_weight['cwd'] * d_loss_cw).unsqueeze(0))).detach()