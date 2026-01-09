# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES.
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
def align(*variables: Variable) -> list[Constraint]: ...
@overload
def align(*variables: str) -> list[DeferredConstraint]: ...
@overload
def broadcast(
    variable: Variable, this_name_does_not_exist: Collection[int] = ...
) -> list[Constraint]: ...
@overload
def broadcast(
    variable: Variable, *rest: Variable | tuple[Variable, Collection[int]]
) -> list[Constraint]: ...
@overload
def broadcast(
    variable: str, this_name_does_not_exist: Collection[int] = ...
) -> list[DeferredConstraint]: ...
@overload
def broadcast(
    variable: str, *rest: str | tuple[str, Collection[int]]
) -> list[DeferredConstraint]: ...
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
