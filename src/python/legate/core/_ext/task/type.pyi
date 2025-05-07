# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Generic, Literal, TypeAlias, TypeVar

from ..._lib.data.physical_array import PhysicalArray
from ..._lib.data.physical_store import PhysicalStore
from ..._lib.task.task_context import TaskContext
from ..._lib.type.types import ReductionOpKind

ParamList: TypeAlias = tuple[str, ...]

UserFunction: TypeAlias = Callable[..., None]

VariantFunction: TypeAlias = Callable[[TaskContext], None]

_T = TypeVar("_T", bound=ReductionOpKind)

# Must be up to date with ReductionOpKind
ADD: TypeAlias = Literal[ReductionOpKind.ADD]
MUL: TypeAlias = Literal[ReductionOpKind.MUL]
MAX: TypeAlias = Literal[ReductionOpKind.MAX]
MIN: TypeAlias = Literal[ReductionOpKind.MIN]
OR: TypeAlias = Literal[ReductionOpKind.OR]
AND: TypeAlias = Literal[ReductionOpKind.AND]
XOR: TypeAlias = Literal[ReductionOpKind.XOR]

class InputStore(PhysicalStore): ...
class OutputStore(PhysicalStore): ...
class ReductionStore(Generic[_T], PhysicalStore): ...
class InputArray(PhysicalArray): ...
class OutputArray(PhysicalArray): ...
class ReductionArray(Generic[_T], PhysicalArray): ...
