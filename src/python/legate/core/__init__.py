# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E402
from __future__ import annotations

import random as _random
import warnings
from typing import TYPE_CHECKING, Any

import numpy as _np  # noqa: ICN001


# We need to import libucx for Python pip wheel builds. This is due to the
# fact that liblegate.so links to librealm-legate.so which links to libucx.so
# and we need to ensure when MPI is used and the wrappers are dlopened that
# they will ideally use the system UCX libraries they were compiled against.
# This is why this must happen so early on in the import process and there
# are still potential issues. We fallback to the bundled UCX libraries if the
# system ones are not found, and only look for the unversioned SOs at this
# point (libucs.so not libucs.so.0) which does not work for all installations.
#
# TODO(cryos, jfaibussowit)
# Implement the same loading logic from libucx on the C++ side, and remove
# this workaround.
def _maybe_import_ucx_module() -> Any:
    import os

    from ..install_info import wheel_build

    if not wheel_build:
        return None

    # Prefer wheels libraries that should load a consistent set of libraries.
    #
    # See https://github.com/rapidsai/ucx-wheels/blob/main/python/libucx/libucx/load.py#L55
    # for the environment variable check and logic for library loading.
    try:
        import libucx  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return None

    # The handles are returned here in order to ensure the libraries are
    # loaded for the duration of execution.
    return libucx.load_library()


_libucx = _maybe_import_ucx_module()

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
