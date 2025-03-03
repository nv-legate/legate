# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Collection
from enum import IntEnum, unique
from typing import Any, cast, overload

from ..utilities.unconstructable import Unconstructable

class Variable(Unconstructable): ...
class Constraint(Unconstructable): ...

class DeferredConstraint:
    @property
    def args(self) -> tuple[Any, ...]: ...

@unique
class ImageComputationHint(IntEnum):
    NO_HINT = cast(int, ...)
    MIN_MAX = cast(int, ...)
    FIRST_LAST = cast(int, ...)

@overload
def align(lhs: Variable, rhs: Variable) -> Constraint: ...
@overload
def align(lhs: str, rhs: str) -> DeferredConstraint: ...
@overload
def broadcast(
    variable: Variable, axes: Collection[int] = ...
) -> Constraint: ...
@overload
def broadcast(
    variable: str, axes: Collection[int] = ...
) -> DeferredConstraint: ...
@overload
def image(
    var_function: Variable,
    var_range: Variable,
    hint: ImageComputationHint = ...,
) -> Constraint: ...
@overload
def image(
    var_function: str, var_range: str, hint: ImageComputationHint = ...
) -> DeferredConstraint: ...
@overload
def scale(
    factors: tuple[int, ...], var_smaller: Variable, var_bigger: Variable
) -> Constraint: ...
@overload
def scale(
    factors: tuple[int, ...], var_smaller: str, var_bigger: str
) -> DeferredConstraint: ...
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
) -> DeferredConstraint: ...
