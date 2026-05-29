# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from os import PathLike as os_PathLike
from pathlib import Path
from typing import TypeAlias

from .....data_interface import LogicalStoreLike
from ....data.logical_store import LogicalStore
from ....data.shape import Shape
from ....type.types import Type

Pathlike: TypeAlias = str | os_PathLike[str] | Path
Shapelike: TypeAlias = Shape | Sequence[int]

def from_file(path: Pathlike, store_type: Type) -> LogicalStore: ...
def to_file(path: Pathlike, store: LogicalStoreLike) -> None: ...
def from_tiles(
    path: Pathlike,
    shape: Shapelike,
    store_type: Type,
    tile_shape: tuple[int, ...],
    tile_start: tuple[int, ...] | None = None,
) -> LogicalStore: ...
def to_tiles(
    path: Pathlike,
    store: LogicalStore,
    tile_shape: tuple[int, ...],
    tile_start: tuple[int, ...] | None = None,
) -> None: ...
def from_tiles_by_offsets(
    path: Pathlike,
    shape: Shapelike,
    type: Type,  # noqa: A002
    offsets: tuple[int, ...],
    tile_shape: tuple[int, ...] | None = None,
) -> LogicalStore: ...
