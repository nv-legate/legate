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
from __future__ import annotations

from typing import Callable, Literal, TypeAlias

from ..._lib.data.physical_array import PhysicalArray
from ..._lib.data.physical_store import PhysicalStore
from ..._lib.mapping.mapping import TaskTarget
from ..._lib.task.task_context import TaskContext

SignatureMapping: TypeAlias = dict[str, type]

ParamList: TypeAlias = tuple[str, ...]

UserFunction: TypeAlias = Callable[..., None]

VariantFunction: TypeAlias = Callable[[TaskContext], None]

VariantKind: TypeAlias = Literal["cpu", "gpu", "omp"]

VariantList: TypeAlias = tuple[VariantKind, ...]

VariantMapping: TypeAlias = dict[TaskTarget, UserFunction | None]

class InputStore(PhysicalStore): ...
class OutputStore(PhysicalStore): ...
class ReductionStore(PhysicalStore): ...
class InputArray(PhysicalArray): ...
class OutputArray(PhysicalArray): ...
