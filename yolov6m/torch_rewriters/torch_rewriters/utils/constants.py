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

# Copyright (c) OpenMMLab. All rights reserved.
from enum import Enum


class AdvancedEnum(Enum):
    """Define an enumeration class."""

    @classmethod
    def get(cls, value):
        """Get the key through a value."""
        for k in cls:
            if k.value == value:
                return k

        raise KeyError(f'Cannot get key by value "{value}" of {cls}')


class IR(AdvancedEnum):
    """Define intermediate representation enumerations."""
    ONNX = 'onnx'
    TORCHSCRIPT = 'torchscript'
    XMODEL = 'xmodel'
    DEFAULT = 'default'


class Backend(AdvancedEnum):
    """Define backend enumerations."""
    PYTORCH = 'pytorch'
    TENSORRT = 'tensorrt'
    ONNXRUNTIME = 'onnxruntime'
    PPLNN = 'pplnn'
    NCNN = 'ncnn'
    SNPE = 'snpe'
    OPENVINO = 'openvino'
    SDK = 'sdk'
    TORCHSCRIPT = 'torchscript'
    RKNN = 'rknn'
    ASCEND = 'ascend'
    COREML = 'coreml'
    NNDCT = 'nndct'
    DEFAULT = 'default'
