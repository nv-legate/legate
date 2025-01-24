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

from collections.abc import Sequence
from os import PathLike as os_PathLike
from pathlib import Path
from typing import TypeAlias

from ....data.logical_array import LogicalArray
from ....data.shape import Shape
from ....type.type_info import Type

Pathlike: TypeAlias = str | os_PathLike[str] | Path
Shapelike: TypeAlias = Shape | Sequence[int]

def from_file(path: Pathlike, array_type: Type) -> LogicalArray: ...
def to_file(path: Pathlike, array: LogicalArray) -> None: ...
def from_tiles(
    path: Pathlike,
    shape: Shapelike,
    array_type: Type,
    tile_shape: tuple[int, ...],
    tile_start: tuple[int, ...] | None = None,
) -> LogicalArray: ...
def to_tiles(
    path: Pathlike,
    array: LogicalArray,
    tile_shape: tuple[int, ...],
    tile_start: tuple[int, ...] | None = None,
) -> None: ...
def from_tiles_by_offsets(
    path: Pathlike,
    shape: Shapelike,
    type: Type,  # noqa: A002
    offsets: tuple[int, ...],
    tile_shape: tuple[int, ...] | None = None,
) -> LogicalArray: ...
