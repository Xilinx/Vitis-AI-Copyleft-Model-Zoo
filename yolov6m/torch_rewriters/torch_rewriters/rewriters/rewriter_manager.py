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
from typing import Dict

import torch.nn as nn

from torch_rewriters.utils.constants import IR, Backend
from .function_rewriter import FunctionRewriter
from .module_rewriter import ModuleRewriter
from .rewriter_utils import collect_env
from .symbolic_rewriter import SymbolicRewriter


class RewriterManager:
    """The rewrite manager that manages some rewriters."""

    def __init__(self):
        self.module_rewriter = ModuleRewriter()
        self.function_rewriter = FunctionRewriter()
        self.symbolic_rewriter = SymbolicRewriter()


REWRITER_MANAGER = RewriterManager()

MODULE_REWRITER = REWRITER_MANAGER.module_rewriter
FUNCTION_REWRITER = REWRITER_MANAGER.function_rewriter
SYMBOLIC_REWRITER = REWRITER_MANAGER.symbolic_rewriter


def patch_model(model: nn.Module,
                cfg: Dict,
                backend: str = Backend.DEFAULT.value,
                ir: IR = IR.DEFAULT,
                recursive: bool = True,
                **kwargs) -> nn.Module:
    """Patch the model, replace the modules that can be rewritten. Note that
    the original model will be modified permanently.

    Args:
        model (torch.nn.Module): The model to patch.
        cfg (Dict): Config dictionary of deployment.
        backend (str): The inference engine name.
        ir (IR): The intermeditate representation name.
        recursive (bool): The flag to enable recursive patching.

    Returns:
        nn.Module: THe patched model.

    Examples:
        >>> from torch_rewriters import patch_model
        >>> from torch_rewriters.utils import Backend, IR
        >>> deploy_cfg = {}
        >>> backend = Backend.DEFAULT.value
        >>> ir = IR.ONNX
        >>> patched_model = patch_model(model, deploy_cfg, backend, ir)
    """
    return MODULE_REWRITER.patch_model(model, cfg, backend, ir, recursive,
                                       **kwargs)


class RewriterContext:
    """The rewrite context.

    The context is used to manage the rewrite functions and the backend.

    Args:
        cfg (Dict): Config dictionary of deployment.
        backend (str): The inference engine name.
        ir (IR): The intermeditate representation name.
        rewrite_manager (RewriterManager): An RewriteManager that consists of
            several rewriters

    Examples:
        >>> from torch_rewriters import RewriterContext
        >>> with RewriterContext(cfg, backend='onnxruntime'):
        >>>     # the rewrite has been activated inside the context
        >>>     torch.onnx.export(model, inputs, onnx_file)
    """

    def __init__(self,
                 cfg: Dict = dict(),
                 backend: str = Backend.DEFAULT.value,
                 ir: IR = IR.DEFAULT,
                 rewriter_manager: RewriterManager = REWRITER_MANAGER,
                 **kwargs):
        self._cfg = cfg
        self._kwargs = kwargs
        self._rewriter_manager = rewriter_manager
        self._env = collect_env(Backend.get(backend), ir)

    def enter(self):
        """Call the enter() of rewriters."""
        self._rewriter_manager.function_rewriter.enter(self._cfg, self._env,
                                                       **self._kwargs)
        self._rewriter_manager.symbolic_rewriter.enter(self._cfg, self._env,
                                                       **self._kwargs)

    def exit(self):
        """Call the exit() of rewriters."""
        self._rewriter_manager.function_rewriter.exit()
        self._rewriter_manager.symbolic_rewriter.exit()

    def __enter__(self):
        """Call enter()"""
        self.enter()

    def __exit__(self, type, val, tb):
        """Call exit()"""
        self.exit()