# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from os import PathLike as os_PathLike
from pathlib import Path
from typing import TypeAlias

from ....data.logical_array import LogicalArray
from ....data.shape import Shape
from ....type.types import Type

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
