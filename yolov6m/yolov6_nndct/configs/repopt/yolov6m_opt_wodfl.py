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

# YOLOv6m model
model = dict(
    type='YOLOv6m',
    pretrained=None,
    scales='runs/train/yolov6m_hs/weights/best_ckpt.pt',
    depth_multiple=0.60,  
    width_multiple=0.75,
    backbone=dict(
        type='CSPBepBackbone',
        num_repeats=[1, 6, 12, 18, 6],
        out_channels=[64, 128, 256, 512, 1024],
        csp_e=float(2)/3,
        ),
    neck=dict(
        type='CSPRepPANNeck',
        num_repeats=[12, 12, 12, 12],
        out_channels=[256, 128, 128, 256, 256, 512],
        csp_e=float(2)/3,
        ),
    head=dict(
        type='EffiDeHead',
        in_channels=[128, 256, 512],
        num_layers=3,
        begin_indices=24,
        anchors=1,
        out_indices=[17, 20, 23],
        strides=[8, 16, 32],
        iou_type='giou',
        use_dfl=False,
        reg_max=0, #if use_dfl is False, please set reg_max to 0
        distill_weight={
            'class': 1.0,
            'dfl': 1.0,
        },
    )
)

solver=dict(
    optim='SGD',
    lr_scheduler='Cosine',
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1       
)
 
data_aug = dict(
    hsv_h=0.015,  
    hsv_s=0.7, 
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.9,
    shear=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
)

# Choose Rep-block by the training Mode, choices=["repvgg", "hyper-search", "repopt"]
training_mode='repopt'