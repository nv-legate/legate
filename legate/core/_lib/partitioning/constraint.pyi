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

from collections.abc import Callable, Collection
from typing import Any, TypeAlias, overload

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

@overload
def align(lhs: Variable, rhs: Variable) -> Constraint: ...
@overload
def align(lhs: str, rhs: str) -> ConstraintProxy: ...
@overload
def broadcast(
    variable: Variable, axes: Collection[int] = tuple()
) -> Constraint: ...
@overload
def broadcast(
    variable: str, axes: Collection[int] = tuple()
) -> ConstraintProxy: ...
@overload
def image(var_function: Variable, var_range: Variable) -> Constraint: ...
@overload
def image(var_function: str, var_range: str) -> ConstraintProxy: ...
@overload
def scale(
    factors: tuple[int, ...], var_smaller: Variable, var_bigger: Variable
) -> Constraint: ...
@overload
def scale(
    factors: tuple[int, ...], var_smaller: str, var_bigger: str
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
    var_source: str,
    var_bloat: str,
    low_offsets: tuple[int, ...],
    high_offsets: tuple[int, ...],
) -> ConstraintProxy: ...
