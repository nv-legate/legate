# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Callable, Generic, Literal, TypeAlias, TypeVar

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
