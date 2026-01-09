# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
import sys
import math
from subprocess import CalledProcessError, run
from typing import TYPE_CHECKING

import numpy as np
from numpy.testing import assert_array_equal

import pytest

from legate.core import (
    LogicalArray,
    TaskTarget,
    Type,
    get_legate_runtime,
    types as ty,
)
from legate.core.experimental.io.tile import (
    from_tiles,
    from_tiles_by_offsets,
    to_tiles,
)

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


try:
    run(["modinfo", "nvidia_fs"], check=True)
except (CalledProcessError, FileNotFoundError):
    has_cufile = False
else:
    has_cufile = True


@pytest.mark.skipif(
    not has_cufile
    and get_legate_runtime().machine.preferred_target == TaskTarget.GPU,
    reason="test require nvidia_fs",
)
@pytest.mark.parametrize(
    ("shape", "tile_shape", "offsets", "expected"),
    [
        ((2,), (2,), (0,), [0, 1]),
        ((4,), (2,), (0, 1), [0, 1, 1, 0]),
        ((4, 2), (2, 2), (4, 2), [[0, 0], [0, 0], [2, 3], [0, 0]]),
        ((2, 4), (2, 2), (4, 0), [[0, 0, 0, 1], [0, 0, 4, 5]]),
    ],
    ids=str,
)
def test_read_tiles_by_offset(
    tmp_path: Path,
    shape: tuple[int, ...],
    tile_shape: tuple[int, ...],
    offsets: tuple[int, ...],
    expected: list[int | list[int]],
) -> None:
    a = np.arange(math.prod(shape)).reshape(shape)
    store = get_legate_runtime().create_store_from_buffer(
        Type.from_numpy_dtype(a.dtype), a.shape, a, False
    )
    to_tiles(
        array=LogicalArray.from_store(store),
        path=tmp_path,
        tile_shape=tile_shape,
        tile_start=(0,) * len(tile_shape),
    )
    get_legate_runtime().issue_execution_fence(block=True)

    b = from_tiles_by_offsets(
        path=tmp_path / ".".join(map(str, (0,) * len(tile_shape))),
        shape=store.shape,
        type=store.type,
        offsets=tuple(
            Type.from_numpy_dtype(a.dtype).size * offset for offset in offsets
        ),
        tile_shape=tile_shape,
    )
    get_legate_runtime().issue_execution_fence(block=True)

    arr_exp = np.array(expected)
    # legate may or may not get a clean array here, need to wipe everything
    # we don't expect to read with 0 before comparing
    arr_b = np.asarray(b.get_physical_array())
    arr_b[:][np.where(arr_exp == 0)] = 0
    np.testing.assert_allclose(arr_b, arr_exp)


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
