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

# eval param for different scale

eval_params = dict(
    default = dict(
        img_size=640,
        test_load_size=634,
        letterbox_return_int=True,
        scale_exact=True,
        force_no_pad=True,
        not_infer_on_rect=True,
    ),
    yolov6n = dict(
        img_size=640,
        test_load_size=638,
        letterbox_return_int=True,
        scale_exact=True,
        force_no_pad=True,
        not_infer_on_rect=True,
    ),
    yolov6t = dict(
        img_size=640,
        test_load_size=634,
        letterbox_return_int=True,
        scale_exact=True,
        force_no_pad=True,
        not_infer_on_rect=True,
    ),
    yolov6s = dict(
        img_size=640,
        test_load_size=638,
        letterbox_return_int=True,
        scale_exact=True,
        force_no_pad=True,
        not_infer_on_rect=True,
    ),
    yolov6m = dict(
        img_size=640,
        test_load_size=628,
        letterbox_return_int=True,
        scale_exact=True,
        force_no_pad=True,
        not_infer_on_rect=True,
    ),
    yolov6l = dict(
        img_size=640,
        test_load_size=632,
        letterbox_return_int=True,
        scale_exact=True,
        force_no_pad=True,
        not_infer_on_rect=True,
    ),
    yolov6l_relu = dict(
        img_size=640,
        test_load_size=638,
        letterbox_return_int=True,
        scale_exact=True,
        force_no_pad=True,
        not_infer_on_rect=True,
    )
)