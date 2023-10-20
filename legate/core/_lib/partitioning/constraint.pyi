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

from typing import Iterable, Optional

class Variable:
    def __str__(self) -> str: ...

class Constraint:
    def __str__(self) -> str: ...

def align(lhs: Variable, rhs: Variable) -> Constraint: ...
def broadcast(
    variable: Variable, axes: Optional[Iterable[int]] = None
) -> Constraint: ...
def image(var_function: Variable, var_range: Variable) -> Constraint: ...
def scale(
    factors: tuple[int, ...], var_smaller: Variable, var_bigger: Variable
) -> Constraint: ...
def bloat(
    var_source: Variable,
    var_bloat: Variable,
    low_offsets: tuple[int, ...],
    high_offsets: tuple[int, ...],
) -> Constraint: ...
