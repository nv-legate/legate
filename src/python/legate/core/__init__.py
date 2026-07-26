# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import random as _random
import warnings
from typing import TYPE_CHECKING, Any

import numpy as _np  # noqa: ICN001

from ._lib.data.buffer import Buffer
from ._lib.data.logical_store import LogicalStore, LogicalStorePartition
from ._lib.data.physical_store import PhysicalStore
from ._lib.data.raw_allocation import RawAllocation
from ._lib.data.scalar import Scalar
from ._lib.data.shape import Shape
from ._lib.geometry import Point, Rect
from ._lib.legate_defines import LEGATE_MAX_DIM
from ._lib.mapping.machine import (
    EmptyMachineError,
    Machine,
    ProcessorRange,
    ProcessorSlice,
)
from ._lib.mapping.mapping import DimOrdering, StoreTarget, TaskTarget
from ._lib.operation.projection import SymbolicExpr, constant, dimension
from ._lib.operation.task import AutoTask, ManualTask
from ._lib.partitioning.constraint import (
    Constraint,
    ImageComputationHint,
    Variable,
    align,
    bloat,
    broadcast,
    image,
    min_extents,
    scale,
)
from ._lib.runtime.library import Library
from ._lib.runtime.resource import ResourceConfig
from ._lib.runtime.runtime import (
    Runtime,
    get_legate_runtime,
    get_machine,
    is_running_in_task,
    track_provenance,
)
from ._lib.task.task_config import TaskConfig
from ._lib.task.task_context import TaskContext
from ._lib.task.task_info import TaskInfo
from ._lib.task.variant_options import VariantOptions
from ._lib.tuning.parallel_policy import ParallelPolicy, StreamingMode
from ._lib.tuning.scope import Scope
from ._lib.utilities.detail.dlpack.from_dlpack import from_dlpack
from ._lib.utilities.typedefs import (
    GlobalRedopID,
    LocalRedopID,
    LocalTaskID,
    VariantCode,
)
from .data_interface import LegateDataInterface, as_logical_store
from .types import (
    FixedArrayType,
    ReductionOpKind,
    StructType,
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

__all__ = (
    "LEGATE_MAX_DIM",
    "Annotation",
    "AutoTask",
    "Buffer",
    "Constraint",
    "DimOrdering",
    "EmptyMachineError",
    "FixedArrayType",
    "GlobalRedopID",
    "ImageComputationHint",
    "LegateDataInterface",
    "Library",
    "LocalRedopID",
    "LocalTaskID",
    "LogicalStore",
    "LogicalStorePartition",
    "Machine",
    "ManualTask",
    "PhysicalStore",
    "Point",
    "ProcessorRange",
    "ProcessorSlice",
    "RawAllocation",
    "Rect",
    "ReductionOpKind",
    "ResourceConfig",
    "Runtime",
    "Scalar",
    "Scope",
    "Shape",
    "StoreTarget",
    "StructType",
    "SymbolicExpr",
    "TaskConfig",
    "TaskContext",
    "TaskInfo",
    "TaskTarget",
    "Type",
    "TypeCode",
    "Variable",
    "VariantCode",
    "VariantOptions",
    "align",
    "array_type",
    "binary_type",
    "bloat",
    "bool_",
    "broadcast",
    "complex64",
    "complex128",
    "constant",
    "dimension",
    "float16",
    "float32",
    "float64",
    "get_legate_runtime",
    "get_machine",
    "image",
    "int8",
    "int16",
    "int32",
    "int64",
    "is_running_in_task",
    "min_extents",
    "null_type",
    "scale",
    "string_type",
    "struct_type",
    "track_provenance",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
)
