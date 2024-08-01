# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

from .data_interface import LegateDataInterface, Field, Table
from ._lib.mapping.mapping import StoreTarget, TaskTarget
from ._lib.mapping.machine import (
    EmptyMachineError,
    Machine,
    ProcessorRange,
    ProcessorSlice,
)
from ._lib.runtime.runtime import (
    Runtime,
    get_legate_runtime,
    get_machine,
    track_provenance,
    is_running_in_task,
)
from ._lib.runtime.scope import Scope

get_legate_runtime()
from .utils import Annotation
from ._lib.legate_defines import LEGATE_MAX_DIM
from ._lib.utilities.typedefs import (
    GlobalTaskID,
    LocalTaskID,
    LocalRedopID,
    GlobalRedopID,
)
from ._lib.data.inline_allocation import InlineAllocation
from ._lib.data.logical_array import LogicalArray
from ._lib.data.logical_store import LogicalStore, LogicalStorePartition
from ._lib.data.scalar import Scalar
from ._lib.data.shape import Shape
from ._lib.data.physical_store import PhysicalStore
from ._lib.data.physical_array import PhysicalArray
from ._lib.operation.projection import dimension, constant
from ._lib.operation.task import AutoTask, ManualTask
from ._lib.partitioning.constraint import (
    ImageComputationHint,
    align,
    image,
    bloat,
    broadcast,
    scale,
)
from ._lib.runtime.exception_mode import ExceptionMode
from ._lib.runtime.library import Library
from ._lib.task.task_context import TaskContext
from ._lib.task.task_info import TaskInfo

from .types import (
    ReductionOpKind,
    Type,
    array_type,
    binary_type,
    bool_,
    complex128,
    complex64,
    float16,
    float32,
    float64,
    int16,
    int32,
    int64,
    int8,
    null_type,
    string_type,
    struct_type,
    uint16,
    uint32,
    uint64,
    uint8,
)

import warnings
import numpy as _np
import random as _random
from typing import Any
from .utils import AnyCallable

_np.random.seed(1234)
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
