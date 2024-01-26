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

from collections.abc import Callable, Sequence
from typing import Any, TypeAlias, overload

from ..data.logical_array import LogicalArray
from ..data.logical_store import LogicalStore

class Variable:
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class Constraint:
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class ConstraintProxy:
    def __init__(
        self, func: Callable[..., Constraint], *args: Any
    ) -> None: ...
    @property
    def func(self) -> Callable[..., Constraint]: ...
    @property
    def args(self) -> tuple[Any, ...]: ...

StoreOrArray: TypeAlias = LogicalStore | LogicalArray

@overload
def align(lhs: Variable, rhs: Variable) -> Constraint: ...
@overload
def align(lhs: StoreOrArray, rhs: StoreOrArray) -> ConstraintProxy: ...
@overload
def broadcast(
    variable: Variable, axes: Sequence[int] | None = None
) -> Constraint: ...
@overload
def broadcast(
    variable: StoreOrArray, axes: Sequence[int] | None = None
) -> ConstraintProxy: ...
@overload
def image(var_function: Variable, var_range: Variable) -> Constraint: ...
@overload
def image(
    var_function: StoreOrArray, var_range: StoreOrArray
) -> ConstraintProxy: ...
@overload
def scale(
    factors: tuple[int, ...], var_smaller: Variable, var_bigger: Variable
) -> Constraint: ...
@overload
def scale(
    factors: tuple[int, ...],
    var_smaller: StoreOrArray,
    var_bigger: StoreOrArray,
) -> ConstraintProxy: ...
@overload
def bloat(
    var_source: Variable,
    var_bloat: Variable,
    low_offsets: tuple[int, ...],
    high_offsets: tuple[int, ...],
) -> Constraint: ...
@overload
def bloat(
    var_source: StoreOrArray,
    var_bloat: StoreOrArray,
    low_offsets: tuple[int, ...],
    high_offsets: tuple[int, ...],
) -> ConstraintProxy: ...
