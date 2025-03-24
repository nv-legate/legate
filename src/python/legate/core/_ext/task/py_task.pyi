# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Any

from ..._lib.operation.task import AutoTask
from ..._lib.partitioning.constraint import DeferredConstraint
from ..._lib.runtime.library import Library
from ..._lib.task.task_config import TaskConfig
from ..._lib.task.variant_options import VariantOptions
from ..._lib.utilities.typedefs import LocalTaskID, VariantCode
from .invoker import VariantInvoker
from .type import UserFunction

class PyTask:
    def __init__(
        self,
        *,
        func: UserFunction,
        variants: Sequence[VariantCode],
        constraints: Sequence[DeferredConstraint] | None = None,
        options: TaskConfig | VariantOptions | None = None,
        invoker: VariantInvoker | None = None,
        library: Library | None = None,
        register: bool = True,
    ): ...
    @property
    def registered(self) -> bool: ...
    @property
    def task_id(self) -> LocalTaskID: ...
    @property
    def library(self) -> Library: ...
    def prepare_call(self, *args: Any, **kwargs: Any) -> AutoTask: ...
    def __call__(self, *args: Any, **kwargs: Any) -> None: ...
    def complete_registration(self) -> LocalTaskID: ...
    def cpu_variant(self, func: UserFunction) -> None: ...
    def gpu_variant(self, func: UserFunction) -> None: ...
    def omp_variant(self, func: UserFunction) -> None: ...
