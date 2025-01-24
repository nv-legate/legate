# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import random as _random
import warnings
from typing import TYPE_CHECKING, Any

import numpy as _np  # noqa: ICN001

from ._lib.data.inline_allocation import InlineAllocation
from ._lib.data.logical_array import LogicalArray
from ._lib.data.logical_store import LogicalStore, LogicalStorePartition
from ._lib.data.physical_array import PhysicalArray
from ._lib.data.physical_store import PhysicalStore
from ._lib.data.scalar import Scalar
from ._lib.data.shape import Shape
from ._lib.legate_defines import LEGATE_MAX_DIM
from ._lib.mapping.machine import (
    EmptyMachineError,
    Machine,
    ProcessorRange,
    ProcessorSlice,
)
from ._lib.mapping.mapping import StoreTarget, TaskTarget
from ._lib.operation.projection import constant, dimension
from ._lib.operation.task import AutoTask, ManualTask
from ._lib.partitioning.constraint import (
    ImageComputationHint,
    align,
    bloat,
    broadcast,
    image,
    scale,
)
from ._lib.runtime.exception_mode import ExceptionMode
from ._lib.runtime.library import Library
from ._lib.runtime.runtime import (
    ProfileRange,
    Runtime,
    get_legate_runtime,
    get_machine,
    is_running_in_task,
    track_provenance,
)
from ._lib.runtime.scope import Scope
from ._lib.task.task_context import TaskContext
from ._lib.task.task_info import TaskInfo
from ._lib.utilities.typedefs import (
    GlobalRedopID,
    GlobalTaskID,
    LocalRedopID,
    LocalTaskID,
    VariantCode,
)
from .data_interface import Field, LegateDataInterface, Table
from .types import (
    ReductionOpKind,
    Type,
    TypeCode,
    array_type,
    binary_type,
    bool_,
    complex64,
    complex128,
    float16,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    null_type,
    string_type,
    struct_type,
    uint8,
    uint16,
    uint32,
    uint64,
)
from .utils import Annotation

if TYPE_CHECKING:
    from .utils import AnyCallable

_np.random.seed(1234)  # noqa: NPY002
_random.seed(1234)


def _warn_seed(func: AnyCallable) -> AnyCallable:
    def wrapper(*args: Any, **kw: Any) -> Any:
        msg = """
        Seeding the random number generator with a non-constant value
        inside Legate can lead to undefined behavior and/or errors when
        the program is executed with multiple ranks."""
        warnings.warn(msg, Warning, stacklevel=2)
        return func(*args, **kw)

    return wrapper


_np.random.seed = _warn_seed(_np.random.seed)
_random.seed = _warn_seed(_random.seed)

get_legate_runtime()  # Starts the runtime
