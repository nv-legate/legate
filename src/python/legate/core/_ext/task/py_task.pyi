# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from collections.abc import Sequence
from typing import Any, Final

from ..._lib.operation.task import AutoTask
from ..._lib.partitioning.constraint import ConstraintProxy
from ..._lib.runtime.library import Library
from ..._lib.utilities.typedefs import LocalTaskID
from .invoker import VariantInvoker
from .type import UserFunction, VariantList

class PyTask:
    UNREGISTERED_ID: Final = ...

    def __init__(
        self,
        *,
        func: UserFunction,
        variants: VariantList,
        constraints: Sequence[ConstraintProxy] | None = None,
        throws_exception: bool = False,
        has_side_effect: bool = False,
        invoker: VariantInvoker | None = None,
        library: Library | None = None,
        register: bool = True,
    ): ...
    @property
    def registered(self) -> bool: ...
    @property
    def task_id(self) -> LocalTaskID: ...
    def prepare_call(self, *args: Any, **kwargs: Any) -> AutoTask: ...
    def __call__(self, *args: Any, **kwargs: Any) -> None: ...
    def complete_registration(self) -> LocalTaskID: ...
    def cpu_variant(self, func: UserFunction) -> None: ...
    def gpu_variant(self, func: UserFunction) -> None: ...
    def omp_variant(self, func: UserFunction) -> None: ...
