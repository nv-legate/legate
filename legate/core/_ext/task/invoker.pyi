# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
from __future__ import annotations

from inspect import Signature
from typing import Any

from ..._lib.operation.task import AutoTask
from ..._lib.partitioning.constraint import ConstraintProxy
from ..._lib.task.task_context import TaskContext
from .type import ConstraintSet, ParamList, UserFunction

class VariantInvoker:
    def __init__(self, func: UserFunction) -> None: ...
    @property
    def inputs(self) -> ParamList: ...
    @property
    def outputs(self) -> ParamList: ...
    @property
    def reductions(self) -> ParamList: ...
    @property
    def scalars(self) -> ParamList: ...
    @property
    def signature(self) -> Signature: ...
    def prepare_call(
        self,
        task: AutoTask,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        constraints: ConstraintSet | None = None,
    ) -> None: ...
    def __call__(self, ctx: TaskContext, func: UserFunction) -> None: ...
    def valid_signature(self, func: UserFunction) -> bool: ...
    def validate_signature(self, func: UserFunction) -> None: ...
