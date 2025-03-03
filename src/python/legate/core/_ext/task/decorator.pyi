# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Callable, overload

from ..._lib.partitioning.constraint import DeferredConstraint
from ..._lib.utilities.typedefs import VariantCode
from .py_task import PyTask
from .type import UserFunction

@overload
def task(func: UserFunction) -> PyTask: ...
@overload
def task(
    *,
    variants: tuple[VariantCode, ...] = ...,
    constraints: Sequence[DeferredConstraint] | None = None,
    throws_exception: bool = False,
    has_side_effect: bool = False,
    register: bool = True,
) -> Callable[[UserFunction], PyTask]: ...
