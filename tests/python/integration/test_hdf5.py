# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
import sys
import math
from typing import TYPE_CHECKING

import h5py  # type: ignore # noqa: PGH003
import numpy as np
from numpy.testing import assert_array_equal

import pytest

from legate.core import get_legate_runtime
from legate.io.hdf5 import from_file, kerchunk_read

if TYPE_CHECKING:
    from pathlib import Path

shape_chunks = (
    "shape,chunks",
    [
        ((2,), (2,)),
        ((5,), (2,)),
        ((4, 2), (2, 2)),
        ((2, 4), (2, 2)),
        ((2, 3), (2, 2)),
        ((5, 4, 3, 2), (2, 2, 2, 2)),
    ],
)

# Called for effect to start legate runtime
get_legate_runtime()


@pytest.mark.parametrize(*shape_chunks)
@pytest.mark.parametrize("dtype", ["u1", "u8", "f8"])
def test_array_read(
    tmp_path: Path, shape: tuple[int, ...], chunks: tuple[int, ...], dtype: str
) -> None:
    filename = tmp_path / "test-file.hdf5"
    a = np.arange(math.prod(shape), dtype=dtype).reshape(shape)
    with h5py.File(filename, "w") as f:
        f.create_dataset("dataset", chunks=chunks, data=a)

    b = from_file(filename, dataset_name="dataset")
    get_legate_runtime().issue_execution_fence(block=True)
    assert_array_equal(a, np.asarray(b.get_physical_array()))


def test_array_read_scalar(tmp_path: Path) -> None:
    filename = tmp_path / "test-file.hdf5"

    a = np.asarray(42)
    with h5py.File(filename, "w") as f:
        f.create_dataset("dataset", data=a)

    b = from_file(filename, dataset_name="dataset")
    get_legate_runtime().issue_execution_fence(block=True)
    assert_array_equal(a, np.asarray(b.get_physical_array()))


def test_array_read_of_group(tmp_path: Path) -> None:
    fname = "data.hdf5"
    a = np.arange(10, dtype="i4").reshape((1, 10))
    with h5py.File(tmp_path / fname, "w") as f:
        root = f.create_group("root")
        root.create_dataset("dataset", data=a)
    b = from_file(tmp_path / fname, dataset_name="root/dataset")
    assert_array_equal(a, np.asarray(b.get_physical_array()))

    with pytest.raises(
        ValueError,
        match="Dataset 'root' does not exist in .*" + re.escape(fname),
    ):
        from_file(tmp_path / fname, dataset_name="root")


def test_array_read_virtual_dataset(tmp_path: Path) -> None:
    # Write a hdf5 file that contains a virtual dataset
    a = np.arange(10, dtype="i4").reshape((1, 10))
    with h5py.File(tmp_path / "data.hdf5", "w") as f:
        f.create_dataset("dataset", data=a)
    layout = h5py.VirtualLayout(shape=a.shape, dtype=a.dtype)
    layout[0] = h5py.VirtualSource(
        tmp_path / "data.hdf5", "dataset", shape=a.shape
    )
    with h5py.File(tmp_path / "virtual-data.hdf5", "w") as f:
        root = f.create_group("root")
        root.create_virtual_dataset("virtual-dataset", layout)

    b = from_file(
        tmp_path / "virtual-data.hdf5", dataset_name="root/virtual-dataset"
    )
    assert_array_equal(a, np.asarray(b.get_physical_array()))


@pytest.mark.parametrize("dtype", ["float16", "complex64", "complex128"])
def test_array_read_unsupported_dtype(tmp_path: Path, dtype: str) -> None:
    filename = tmp_path / "test-file.hdf5"
    a = np.arange(10, dtype=dtype).reshape(2, 5)
    with h5py.File(filename, "w") as f:
        f.create_dataset("dataset", data=a)

    with pytest.raises(
        ValueError, match=r"unsupported (floating point size|HDF5 datatype).*"
    ):
        from_file(filename, dataset_name="dataset")


@pytest.mark.parametrize(*shape_chunks)
@pytest.mark.parametrize("dtype", ["u1", "u8", "f8"])
def test_kerchunk_read(
    tmp_path: Path, shape: tuple[int, ...], chunks: tuple[int, ...], dtype: str
) -> None:
    filename = tmp_path / "test-file.hdf5"
    a = np.arange(math.prod(shape), dtype=dtype).reshape(shape)
    with h5py.File(filename, "w") as f:
        f.create_dataset("dataset", chunks=chunks, data=a)

    barr = kerchunk_read(filename, dataset_name="dataset")
    b = np.asarray(barr.get_physical_array())
    assert_array_equal(a, b)

    # Check the `out` argument when shape is divisible by chunk
    if all(s % c == 0 for s, c in zip(shape, chunks, strict=True)):
        cc = kerchunk_read(filename, dataset_name="dataset")
        assert_array_equal(a, np.asarray(cc.get_physical_array()))


def test_kerchunk_read_in_group(tmp_path: Path) -> None:
    filename = tmp_path / "test-file.hdf5"
    a = np.arange(10).reshape((2, 5))
    with h5py.File(filename, "w") as f:
        root = f.create_group("root")
        root.create_dataset("dataset", data=a)

    b = kerchunk_read(filename, dataset_name="root/dataset")
    assert_array_equal(a, np.asarray(b.get_physical_array()))


def test_kerchunk_read_of_a_virtual_dataset(tmp_path: Path) -> None:
    # Write a hdf5 file that contains a virtual dataset
    a = np.arange(10, dtype="i4").reshape((1, 10))
    with h5py.File(tmp_path / "data.hdf5", "w") as f:
        f.create_dataset("dataset", data=a)
    layout = h5py.VirtualLayout(shape=a.shape, dtype=a.dtype)
    layout[0] = h5py.VirtualSource(
        tmp_path / "data.hdf5", "dataset", shape=a.shape
    )
    with h5py.File(tmp_path / "virtual-data.hdf5", "w") as f:
        root = f.create_group("root")
        root.create_virtual_dataset("virtual-dataset", layout)

    # Let's try to read the virtual dataset
    with pytest.raises(
        NotImplementedError,
        match="Virtual dataset isn't supported: root/virtual-dataset",
    ):
        kerchunk_read(
            tmp_path / "virtual-data.hdf5", dataset_name="root/virtual-dataset"
        )


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
