# Copyright 2023 NVIDIA Corporation
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

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import zarr  # type: ignore
from numpy.testing import assert_array_equal

from legate.core import LogicalArray, Type, get_legate_runtime
from legate.core.experimental.io.zarr import read_array, write_array

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


@pytest.mark.parametrize(*shape_chunks)
@pytest.mark.parametrize("dtype", ["u1", "u8", "f8"])
def test_write_array(
    tmp_path: Path, shape: tuple[int, ...], chunks: tuple[int, ...], dtype: str
) -> None:
    """Test write of a Zarr array"""
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
    """Test read of a Zarr array"""
    a = np.arange(math.prod(shape), dtype=dtype).reshape(shape)
    zarr.open_array(
        tmp_path, mode="w", shape=shape, chunks=chunks, compressor=None
    )[...] = a

    array = read_array(dirpath=tmp_path)
    b = np.asarray(array.get_physical_array())
    assert_array_equal(a, b)


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
