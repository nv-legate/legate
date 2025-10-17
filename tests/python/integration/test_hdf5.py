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

from legate.core import (
    LogicalArray,
    ParallelPolicy,
    Scope,
    StreamingMode,
    Type,
    get_legate_runtime,
    types as ty,
)
from legate.io.hdf5 import from_file, from_file_batched, to_file

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


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_array_read_unsupported_dtype(tmp_path: Path, dtype: str) -> None:
    filename = tmp_path / "test-file.hdf5"
    a = np.arange(10, dtype=dtype).reshape(2, 5)
    with h5py.File(filename, "w") as f:
        f.create_dataset("dataset", data=a)

    with pytest.raises(
        ValueError, match=r"unsupported (floating point size|HDF5 datatype).*"
    ):
        from_file(filename, dataset_name="dataset")


@pytest.mark.parametrize(
    "dtype", [ty.int8, ty.uint16, ty.int32, ty.float64, ty.float32, ty.uint64]
)
@pytest.mark.parametrize(*shape_chunks)
def test_from_file_batched(
    tmp_path: Path,
    dtype: Type,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
) -> None:
    filename = tmp_path / "test-file.h5"
    data = np.arange(math.prod(shape), dtype=dtype.to_numpy_dtype()).reshape(
        *shape
    )
    dataset_name = "foo"
    with h5py.File(filename, "w") as f:
        f[dataset_name] = data

    for arr, offsets in from_file_batched(filename, dataset_name, chunks):
        assert isinstance(arr, LogicalArray)
        assert arr.type == dtype

        phys = arr.get_physical_array()
        arr_np = np.asarray(phys)

        # Shape is not necessarily divisible, so some batches won't have
        # exactly the same shape, but they should the same dimensions
        assert len(arr_np.shape) == len(chunks)
        assert arr_np.shape <= chunks

        slices = tuple(
            slice(o, o + s) for o, s in zip(offsets, chunks, strict=True)
        )
        assert_array_equal(arr_np, data[slices])


@pytest.mark.parametrize("chunk_size", ((0, 1, 2), (-1, 2, 3)))
def test_from_file_batched_invalid_chunk_size(
    chunk_size: tuple[int, ...],
) -> None:
    m = re.escape(f"Invalid chunk size ({chunk_size}), must be >0")
    with pytest.raises(ValueError, match=m):  # noqa: PT012
        # Python generators are fully lazy, executing the function body only
        # once you start iterating the generator. So we need to iterate it in
        # order to trigger the error checks.
        for _ in from_file_batched("foo.h5", "foo", chunk_size):
            pytest.fail("Should never actually iterate the generator")


def test_from_file_batched_invalid_chunk_dim(tmp_path: Path) -> None:
    chunk_size = (2, 2)
    shape = (*chunk_size, 2)
    data = np.ones(shape)
    dataset_name = "foo"
    filename = tmp_path / "test-file.h5"

    with h5py.File(filename, "w") as f:
        f[dataset_name] = data

    m = re.escape(
        f"Dimensions of chunks ({len(chunk_size)}) must match "
        f"dimension of dataset ({len(shape)})."
    )
    with pytest.raises(ValueError, match=m):  # noqa: PT012
        # Python generators are fully lazy, executing the function body only
        # once you start iterating the generator. So we need to iterate it in
        # order to trigger the error checks.
        for _ in from_file_batched(filename, dataset_name, chunk_size):
            pytest.fail("Should never actually iterate the generator")


@pytest.mark.parametrize("shape", [(1,), (2, 2), (3, 4, 5)])
@pytest.mark.parametrize("dtype", [ty.int32, ty.int16, ty.float32, ty.float16])
def test_array_write(
    tmp_path: Path, shape: tuple[int, ...], dtype: Type
) -> None:
    runtime = get_legate_runtime()

    filename = tmp_path / "test-file.hdf5"
    dataset_name = "foo"

    with Scope(
        parallel_policy=ParallelPolicy(
            streaming_mode=StreamingMode.RELAXED, overdecompose_factor=8
        )
    ):
        array = runtime.create_array(dtype=dtype, shape=shape)
        runtime.issue_fill(array, 1)

        to_file(array=array, path=filename, dataset_name=dataset_name)
        # del is deliberate here, we need the array to be destroyed and emit
        # the discard inside the streaming scope.
        del array

    runtime.issue_execution_fence(block=True)


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
