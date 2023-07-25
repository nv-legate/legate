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

from .env import LEGATE_MAX_DIM, LEGATE_MAX_FIELDS
from .field import FieldID
from .future import Future, FutureMap
from .geometry import Point, Rect, Domain
from .operation import (
    Acquire,
    Attach,
    Copy,
    Detach,
    Dispatchable,
    Fill,
    IndexAttach,
    IndexCopy,
    IndexDetach,
    IndexFill,
    Release,
    InlineMapping,
)
from .partition import IndexPartition, Partition
from .partition_functor import (
    PartitionFunctor,
    PartitionByRestriction,
    PartitionByImage,
    PartitionByImageRange,
    EqualPartition,
    PartitionByWeights,
    PartitionByDomain,
)
from .region import Region, OutputRegion, PhysicalRegion
from .space import IndexSpace, FieldSpace
from .task import ArgumentMap, Fence, Task, IndexTask
from .transform import Transform, AffineTransform
from .util import (
    dispatch,
    BufferBuilder,
    ExternalResources,
    FieldListLike,
    legate_task_preamble,
    legate_task_postamble,
    legate_task_progress,
)

__all__ = (
    "Acquire",
    "AffineTransform",
    "ArgumentMap",
    "Attach",
    "BufferBuilder",
    "Copy",
    "Detach",
    "dispatch",
    "Domain",
    "EqualPartition",
    "ExternalResources",
    "Fence",
    "FieldID",
    "FieldListLike",
    "FieldSpace",
    "Fill",
    "Future",
    "FutureMap",
    "IndexAttach",
    "IndexCopy",
    "IndexDetach",
    "IndexFill",
    "IndexPartition",
    "IndexSpace",
    "IndexTask",
    "InlineMapping",
    "OutputRegion",
    "Partition",
    "PartitionByDomain",
    "PartitionByImage",
    "PartitionByImageRange",
    "PartitionByRestriction",
    "PartitionByWeights",
    "PartitionFunctor",
    "PhysicalRegion",
    "Point",
    "Rect",
    "Region",
    "Release",
    "Task",
    "Transform",
    "legate_task_preamble",
    "legate_task_postamble",
    "legate_task_progress",
    "LEGATE_MAX_DIM",
    "LEGATE_MAX_FIELDS",
)
