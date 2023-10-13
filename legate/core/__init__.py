# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

from legion_cffi import is_legion_python, ffi, lib as legion

if is_legion_python == False:
    from legion_top import (
        legion_canonical_python_main,
        legion_canonical_python_cleanup,
    )
    from ..driver.main import prepare_driver, CanonicalDriver
    import atexit, os, shlex, sys

    argv = ["legate"] + shlex.split(os.environ.get("LEGATE_CONFIG", ""))

    driver = prepare_driver(argv, CanonicalDriver)

    if driver.dry_run:
        sys.exit(0)

    os.environ.update(driver.env)

    legion_canonical_python_main(driver.cmd)
    atexit.register(legion_canonical_python_cleanup)

from ._legion import (
    LEGATE_MAX_DIM,
    LEGATE_MAX_FIELDS,
    Point,
    Rect,
    Domain,
    Transform,
    AffineTransform,
    IndexAttach,
    IndexDetach,
    IndexSpace,
    PartitionFunctor,
    PartitionByDomain,
    PartitionByRestriction,
    PartitionByImage,
    PartitionByImageRange,
    EqualPartition,
    PartitionByWeights,
    IndexPartition,
    FieldSpace,
    FieldID,
    Region,
    Partition,
    Fill,
    IndexFill,
    Copy,
    IndexCopy,
    Attach,
    Detach,
    Acquire,
    Release,
    Future,
    OutputRegion,
    PhysicalRegion,
    InlineMapping,
    Task,
    FutureMap,
    IndexTask,
    Fence,
    ArgumentMap,
    BufferBuilder,
    legate_task_preamble,
    legate_task_progress,
    legate_task_postamble,
)

# Import select types for Legate library construction
from .allocation import DistributedAllocation
from .legate import (
    Array,
    Field,
    Library,
    Table,
)
from .machine import (
    EmptyMachineError,
    Machine,
    ProcessorKind,
    ProcessorRange,
    ProcessorSlice,
)
from .runtime import (
    Annotation,
    get_legate_runtime,
    get_legion_context,
    get_legion_runtime,
    get_machine,
    legate_add_library,
    track_provenance,
)
from .store import Store

from .types import (
    array_type,
    bool_,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float16,
    float32,
    float64,
    complex64,
    complex128,
    struct_type,
    Dtype,
    ReductionOp,
)
from .io import CustomSplit, TiledSplit, ingest

import warnings
import numpy as _np
import random as _random
from typing import Any
from .runtime import AnyCallable

_np.random.seed(1234)
_random.seed(1234)


def _warn_seed(func: AnyCallable) -> AnyCallable:
    def wrapper(*args: Any, **kw: Any) -> Any:
        msg = """
        Seeding the random number generator with a non-constant value 
        inside Legate can lead to undefined behavior and/or errors when 
        the program is executed with multiple ranks."""
        warnings.warn(msg, Warning)
        return func(*args, **kw)

    return wrapper


_np.random.seed = _warn_seed(_np.random.seed)
_random.seed = _warn_seed(_random.seed)
