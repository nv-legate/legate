# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
import sys
import math
from pathlib import Path

import h5py  # type: ignore # noqa: PGH003
import numpy as np
from numpy.testing import assert_array_equal

import pytest

from legate.core import (
    Field,
    LogicalArray,
    ParallelPolicy,
    Scope,
    StreamingMode,
    Table,
    Type,
    VariantCode,
    get_legate_runtime,
    types as ty,
)
from legate.core.task import InputStore, task
from legate.io.hdf5 import from_file, from_file_batched, to_file


@task(variants=(VariantCode.CPU,))
def create_hdf5_file_task(
    data_store: InputStore,
    filename: str,
    dataset_name: str,
    chunks: tuple[int, ...],
    is_scalar: bool,
) -> None:
    """Task to create an HDF5 file with a dataset (CPU-only)."""
    data = np.asarray(data_store.get_inline_allocation())

    if is_scalar:
        data = data.reshape(())
    chunks_arg = tuple(chunks) if chunks[0] > 0 else None
    with h5py.File(filename, "w") as f:
        f.create_dataset(dataset_name, chunks=chunks_arg, data=data)


@task(variants=(VariantCode.CPU,))
def create_hdf5_with_group_task(
    data_store: InputStore, filename: str, group_name: str, dataset_name: str
) -> None:
    """Task to create an HDF5 file with a group and dataset (CPU-only)."""
    data = np.asarray(data_store.get_inline_allocation())
    with h5py.File(filename, "w") as f:
        group = f.create_group(group_name)
        group.create_dataset(dataset_name, data=data)


@task(variants=(VariantCode.CPU,))
def create_hdf5_virtual_dataset_task(
    data_store: InputStore, source_filename: str, virtual_filename: str
) -> None:
    """Task to create HDF5 files with a virtual dataset (CPU-only)."""
    data = np.asarray(data_store.get_inline_allocation())
    with h5py.File(source_filename, "w") as f:
        f.create_dataset("dataset", data=data)
    # Create virtual dataset
    layout = h5py.VirtualLayout(shape=data.shape, dtype=data.dtype)
    layout[0] = h5py.VirtualSource(
        source_filename, "dataset", shape=data.shape
    )
    with h5py.File(virtual_filename, "w") as f:
        root = f.create_group("root")
        root.create_virtual_dataset("virtual-dataset", layout)


def create_hdf5_file(
    filename: Path,
    dataset_name: str,
    data: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
    chunks: tuple[int, ...] | None = None,
) -> None:
    """Create an HDF5 file using a single-instance task."""
    filename = Path(filename)
    runtime = get_legate_runtime()

    is_scalar = data.ndim == 0
    if is_scalar:
        data_for_store = data.reshape((1,))
    else:
        data_for_store = data.reshape(data.shape)  # ensure consistent type

    store = runtime.create_store_from_buffer(
        Type.from_numpy_dtype(data_for_store.dtype),
        data_for_store.shape,
        data_for_store,
        read_only=True,
    )

    manual_task = runtime.create_manual_task(
        create_hdf5_file_task.library, create_hdf5_file_task.task_id, (1,)
    )
    manual_task.add_input(store)
    manual_task.add_scalar_arg(str(filename), ty.string_type)
    manual_task.add_scalar_arg(dataset_name, ty.string_type)
    # Use (-1,) as sentinel for None chunks
    manual_task.add_scalar_arg(chunks if chunks else (-1,), (ty.int64,))
    # Pass flag indicating if original data was scalar
    manual_task.add_scalar_arg(is_scalar, ty.bool_)
    manual_task.execute()
    runtime.issue_execution_fence(block=True)


def create_hdf5_with_group(
    filename: str,
    group_name: str,
    dataset_name: str,
    data: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
) -> None:
    """Create an HDF5 file with a group using a single-instance task."""
    runtime = get_legate_runtime()

    # Create a store from the numpy data
    store = runtime.create_store_from_buffer(
        Type.from_numpy_dtype(data.dtype), data.shape, data, read_only=True
    )

    manual_task = runtime.create_manual_task(
        create_hdf5_with_group_task.library,
        create_hdf5_with_group_task.task_id,
        (1,),
    )
    manual_task.add_input(store)
    manual_task.add_scalar_arg(filename, ty.string_type)
    manual_task.add_scalar_arg(group_name, ty.string_type)
    manual_task.add_scalar_arg(dataset_name, ty.string_type)
    manual_task.execute()
    runtime.issue_execution_fence(block=True)


def create_hdf5_virtual_dataset(
    source_filename: str,
    virtual_filename: str,
    data: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
) -> None:
    """Create HDF5 files with virtual dataset using a single-instance task."""
    runtime = get_legate_runtime()

    # Create a store from the numpy data
    store = runtime.create_store_from_buffer(
        Type.from_numpy_dtype(data.dtype), data.shape, data, read_only=True
    )

    manual_task = runtime.create_manual_task(
        create_hdf5_virtual_dataset_task.library,
        create_hdf5_virtual_dataset_task.task_id,
        (1,),
    )
    manual_task.add_input(store)
    manual_task.add_scalar_arg(source_filename, ty.string_type)
    manual_task.add_scalar_arg(virtual_filename, ty.string_type)
    manual_task.execute()
    runtime.issue_execution_fence(block=True)


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

    create_hdf5_file(filename, "dataset", a, chunks)

    b = from_file(filename, dataset_name="dataset")
    get_legate_runtime().issue_execution_fence(block=True)
    assert_array_equal(a, np.asarray(b.get_physical_array()))


def test_array_read_scalar(tmp_path: Path) -> None:
    filename = tmp_path / "test-file.hdf5"

    a = np.asarray(42)
    create_hdf5_file(filename, "dataset", a)

    b = from_file(filename, dataset_name="dataset")
    get_legate_runtime().issue_execution_fence(block=True)
    assert_array_equal(a, np.asarray(b.get_physical_array()))


def test_array_read_of_group(tmp_path: Path) -> None:
    fname = "data.hdf5"
    a = np.arange(10, dtype="i4").reshape((1, 10))
    create_hdf5_with_group(str(tmp_path / fname), "root", "dataset", a)

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
    create_hdf5_virtual_dataset(
        str(tmp_path / "data.hdf5"), str(tmp_path / "virtual-data.hdf5"), a
    )

    b = from_file(
        tmp_path / "virtual-data.hdf5", dataset_name="root/virtual-dataset"
    )
    assert_array_equal(a, np.asarray(b.get_physical_array()))


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_array_read_unsupported_dtype(tmp_path: Path, dtype: str) -> None:
    filename = tmp_path / "test-file.hdf5"
    a = np.arange(10, dtype=dtype).reshape(2, 5)
    create_hdf5_file(filename, "dataset", a)

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
    create_hdf5_file(filename, dataset_name, data)

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

    create_hdf5_file(filename, dataset_name, data)

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


@pytest.mark.parametrize("shape", [(1,), (2, 2), (3, 4, 5)])
def test_array_write_from_data_interface(
    tmp_path: Path, shape: tuple[int, ...]
) -> None:
    runtime = get_legate_runtime()

    filename = tmp_path / "test-file.hdf5"
    dataset_name = "foo"

    with Scope(
        parallel_policy=ParallelPolicy(
            streaming_mode=StreamingMode.RELAXED, overdecompose_factor=8
        )
    ):
        array = runtime.create_array(dtype=ty.int64, shape=shape)
        runtime.issue_fill(array, 1)

        field = Field("foo", dtype=ty.int64)
        table = Table([field], [array])

        to_file(table, path=filename, dataset_name=dataset_name)
        # del is deliberate here, we need the array to be destroyed and emit
        # the discard inside the streaming scope.
        del array

    runtime.issue_execution_fence(block=True)


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
