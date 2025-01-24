# Copyright 2024-2025 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from __future__ import annotations

import re
import sys
import math
from typing import TYPE_CHECKING

import numpy as np
from numpy.testing import assert_array_equal

import pytest

from legate.core import LogicalArray, Type, get_legate_runtime, types as ty
from legate.core.experimental.io.tile import from_tiles, to_tiles

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("shape", "tile_shape"), [((2,), (3,)), ((2, 2), (3, 2)), ((2, 3), (2, 2))]
)
def test_read_write_tiles_error(
    tmp_path: Path, shape: tuple[int, ...], tile_shape: tuple[int, ...]
) -> None:
    match_re = re.escape(
        f"The array shape ({list(shape)}) must be divisible by the "
        f"tile shape ({list(tile_shape)})"
    )
    data = LogicalArray.from_store(
        get_legate_runtime().create_store(ty.int32, shape=shape)
    )
    data.fill(1)
    with pytest.raises(ValueError, match=match_re):
        to_tiles(array=data, path=tmp_path, tile_shape=tile_shape)


@pytest.mark.parametrize(
    ("shape", "tile_shape", "tile_start"),
    [
        ((2,), (2,), (1,)),
        ((4,), (2,), (0,)),
        ((4, 2), (2, 2), (1, 2)),
        ((2, 4), (2, 2), (2, 1)),
    ],
)
def test_read_write_tiles(
    tmp_path: Path,
    shape: tuple[int, ...],
    tile_shape: tuple[int, ...],
    tile_start: tuple[int, ...],
) -> None:
    a = np.arange(math.prod(shape)).reshape(shape)
    store = get_legate_runtime().create_store_from_buffer(
        Type.from_numpy_dtype(a.dtype), a.shape, a, False
    )
    to_tiles(
        array=LogicalArray.from_store(store),
        path=tmp_path,
        tile_shape=tile_shape,
        tile_start=tile_start,
    )
    get_legate_runtime().issue_execution_fence(block=True)
    b = from_tiles(
        path=tmp_path,
        shape=store.shape,
        array_type=store.type,
        tile_shape=tile_shape,
        tile_start=tile_start,
    )
    assert_array_equal(a, np.asarray(b.get_physical_array()))


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
