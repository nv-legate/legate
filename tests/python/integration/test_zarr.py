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

import sys
import math
from typing import TYPE_CHECKING

import zarr  # type: ignore # noqa: PGH003
import numpy as np
from numpy.testing import assert_array_equal

import pytest

from legate.core import LogicalArray, Type, get_legate_runtime
from legate.core.experimental.io.zarr import read_array, write_array

if TYPE_CHECKING:
    from pathlib import Path

shape_chunks = (
    "shape,chunks",
    [
        ((2,), (2,)),
        ((5,), (2,)),
        ((4, 2), (2, 2)),
        ((2, 4), (2, 2)),
        ((2, 3), (3, 2)),
        ((4, 3, 2, 1), (1, 2, 3, 4)),
    ],
)


def is_multi_gpu_ci() -> bool:
    from os import environ  # noqa: PLC0415

    from legate.core import TaskTarget  # noqa: PLC0415

    num_gpu = get_legate_runtime().get_machine().count(target=TaskTarget.GPU)
    return bool(num_gpu > 1 and "CI" in environ)


MULTI_GPU_CI = is_multi_gpu_ci()


@pytest.mark.parametrize(*shape_chunks)
@pytest.mark.parametrize("dtype", ["u1", "u8", "f8"])
@pytest.mark.skipif(
    MULTI_GPU_CI,
    reason=(
        "Intermittent failures in CI for multi-gpu, "
        "see https://github.com/nv-legate/legate.internal/issues/2326"
    ),
)
def test_write_array(
    tmp_path: Path, shape: tuple[int, ...], chunks: tuple[int, ...], dtype: str
) -> None:
    """Test write of a Zarr array."""
    a = np.arange(math.prod(shape), dtype=dtype).reshape(shape)
    store = get_legate_runtime().create_store_from_buffer(
        Type.from_numpy_dtype(a.dtype), a.shape, a, False
    )
    array = LogicalArray.from_store(store)

    write_array(ary=array, dirpath=tmp_path, chunks=chunks)
    get_legate_runtime().issue_execution_fence(block=True)

    b = zarr.open_array(tmp_path, mode="r")
    assert_array_equal(a, b)


@pytest.mark.parametrize(*shape_chunks)
@pytest.mark.parametrize("dtype", ["u1", "u8", "f8"])
def test_read_array(
    tmp_path: Path, shape: tuple[int, ...], chunks: tuple[int, ...], dtype: str
) -> None:
    """Test read of a Zarr array."""
    a = np.arange(math.prod(shape), dtype=dtype).reshape(shape)
    zarr.open_array(
        tmp_path, mode="w", shape=shape, chunks=chunks, compressor=None
    )[...] = a

    array = read_array(dirpath=tmp_path)
    b = np.asarray(array.get_physical_array())
    assert_array_equal(a, b)


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
