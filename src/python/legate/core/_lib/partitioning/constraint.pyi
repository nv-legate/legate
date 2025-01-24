# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
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
from enum import IntEnum, unique
from typing import Any, overload

from ..utilities.unconstructable import Unconstructable

class Variable(Unconstructable): ...
class Constraint(Unconstructable): ...

class ConstraintProxy:
    def __init__(
        self, func: Callable[..., Constraint], *args: Any
    ) -> None: ...
    @property
    def func(self) -> Callable[..., Constraint]: ...
    @property
    def args(self) -> tuple[Any, ...]: ...

@unique
class ImageComputationHint(IntEnum):
    NO_HINT: int
    MIN_MAX: int
    FIRST_LAST: int

@overload
def align(lhs: Variable, rhs: Variable) -> Constraint: ...
@overload
def align(lhs: str, rhs: str) -> ConstraintProxy: ...
@overload
def broadcast(
    variable: Variable, axes: Collection[int] = ...
) -> Constraint: ...
@overload
def broadcast(
    variable: str, axes: Collection[int] = ...
) -> ConstraintProxy: ...
@overload
def image(
    var_function: Variable,
    var_range: Variable,
    hint: ImageComputationHint = ...,
) -> Constraint: ...
@overload
def image(
    var_function: str, var_range: str, hint: ImageComputationHint = ...
) -> ConstraintProxy: ...
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
