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

import re
import sys
import math
from typing import TYPE_CHECKING

import numpy as np
from numpy.testing import assert_array_equal

import pytest

from legate.core import LogicalArray, Type, get_legate_runtime, types as ty
from legate.core.experimental.io.file_handle import FileHandle, OpenFlag

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("size", [1, 10, 100, 1000, 1024, 4096, 4096 * 10])
def test_read_write(tmp_path: Path, size: int) -> None:
    """Test basic read/write."""
    filename = tmp_path / "test-file"
    a = np.arange(math.prod([size])).reshape([size])
    array = LogicalArray.from_store(
        get_legate_runtime().create_store_from_buffer(
            Type.from_numpy_dtype(a.dtype), a.shape, a, False
        )
    )
    f = FileHandle(filename, "w")
    f.write(array)
    assert not f.closed
    get_legate_runtime().issue_execution_fence(block=True)

    # Try to read file opened in write-only mode
    with pytest.raises(
        ValueError, match="Cannot read a file opened with flags"
    ):
        f.read(ty.int32)

    # Close file
    f.close()
    assert f.closed

    # Read file into a new array and compare, not sure why mypy thinks this is
    # unreachable?
    with FileHandle(filename, "r") as f:  # type: ignore[unreachable]
        array2 = f.read(Type.from_numpy_dtype(a.dtype))
    b = np.asarray(array2.get_physical_array())
    assert_array_equal(a, b)


def test_context(tmp_path: Path) -> None:
    """Open a FileHandle in a context."""
    filename = tmp_path / "test-file"
    a = np.arange(math.prod([200])).reshape([200])
    data = LogicalArray.from_store(
        get_legate_runtime().create_store_from_buffer(
            Type.from_numpy_dtype(a.dtype), a.shape, a, False
        )
    )
    with FileHandle(filename, "w+") as f:
        assert not f.closed
        f.write(data)
        get_legate_runtime().issue_execution_fence(block=True)
        out = f.read(data.type)
    b = np.asarray(out.get_physical_array())
    assert_array_equal(a, b)
    assert f.closed


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (0, 10),
        (1, 10),
        (0, 10 * 4096),
        (1, int(1.3 * 4096)),
        (int(2.1 * 4096), int(5.6 * 4096)),
    ],
)
def test_read_write_slices(tmp_path: Path, start: int, end: int) -> None:
    """Read and write different slices."""
    filename = tmp_path / "test-file"
    a = np.arange(math.prod([10 * 4096])).reshape([10 * 4096])  # 10 page-sizes
    a[start:end] = 42
    data = LogicalArray.from_store(
        get_legate_runtime().create_store_from_buffer(
            Type.from_numpy_dtype(a.dtype), a.shape, a, False
        )
    )

    with FileHandle(filename, "w") as f:
        f.write(data)
    get_legate_runtime().issue_execution_fence(block=True)
    with FileHandle(filename, "r") as f:
        out = f.read(data.type)

    b = np.asarray(out.get_physical_array())
    assert_array_equal(a, b)


@pytest.mark.parametrize("flag", ["a", "x", "rx"])
def test_unsupported_flags(tmp_path: Path, flag: OpenFlag) -> None:
    filename = tmp_path / "test-file"
    with pytest.raises(NotImplementedError, match="Unsupported flags"):
        FileHandle(filename, flag)


@pytest.mark.parametrize("flag", ["r", "r+"])
def test_read_nonexistent_file(tmp_path: Path, flag: OpenFlag) -> None:
    filename = tmp_path / "test-file"
    with pytest.raises(FileNotFoundError, match="No such file or directory"):
        FileHandle(filename, flag)


@pytest.mark.parametrize("flag", ["w", "w+", "r", "r+"])
def test_ignore_binary_specifier(tmp_path: Path, flag: OpenFlag) -> None:
    filename = tmp_path / "test-file"
    filename.touch()  # so the "r" flags don't error
    with FileHandle(filename, flag + "b") as f:  # type: ignore[arg-type]
        assert f._flags == flag


def test_bad_array_dims(tmp_path: Path) -> None:
    filename = tmp_path / "test-file"
    a = np.ones([10, 10], dtype="float32")
    data = LogicalArray.from_store(
        get_legate_runtime().create_store_from_buffer(
            Type.from_numpy_dtype(a.dtype), a.shape, a, False
        )
    )
    with (
        FileHandle(filename, "w+") as f,
        pytest.raises(
            ValueError,
            match=re.escape("number of array dimensions must be 1 (have 2)"),
        ),
    ):
        f.write(data)


def test_closed_file(tmp_path: Path) -> None:
    filename = tmp_path / "test-file"
    with FileHandle(filename, "w+") as f:
        pass

    with pytest.raises(RuntimeError, match="file is closed"):
        f.read(ty.int32)


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
