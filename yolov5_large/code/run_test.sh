# Copyright 2023 Advanced Micro Devices, Inc. on behalf of itself and its subsidiaries and affiliates
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




GPU_ID=0

# yolov5 nano weight
# WEIGHTS=./../float/yolov5n_float.pt  
# yolov5 large weight
WEIGHTS=./../float/yolov5l_float.pt
# yolov5s6 weight
# WEIGHTS=./../float/yolov5s6_float.pt
# cd code

echo "[Test mode]"
# img-size 640 for yolov5 nano and yolov5 large
CUDA_VISIBLE_DEVICES=${GPU_ID} python val.py --data data/coco.yaml --img 640 --conf 0.001 --iou 0.65 --weights ${WEIGHTS} --device ${GPU_ID}
# img-size 1280 for yolov5s6
# CUDA_VISIBLE_DEVICES=${GPU_ID} python val.py --data data/coco.yaml --img 1280 --conf 0.001 --iou 0.65 --weights ${WEIGHTS} --device ${GPU_ID}