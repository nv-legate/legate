# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import math
from typing import TYPE_CHECKING

import zarr  # type: ignore # noqa: PGH003
import numpy as np
from numpy.testing import assert_array_equal

import pytest

from legate.core import (
    LogicalArray,
    Table,
    Type,
    VariantCode,
    get_legate_runtime,
    types as ty,
)
from legate.core.experimental.io.zarr import read_array, write_array
from legate.core.task import InputStore, task

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


@task(variants=(VariantCode.CPU,))
def init_test_zarr_task(
    input_store: InputStore,
    dirpath: str,
    array_shape: tuple[int, ...],
    array_dtype: str,
    chunks: tuple[int, ...],
    zarr_format: int,
    use_compressor: bool,
) -> None:
    """Task to initialize test Zarr array (single-instance, CPU-only).

    This task creates a Zarr array and populates it with data from the
    input store. It must run as a single instance to avoid race conditions.
    """
    chunks_arg = tuple(chunks) if chunks[0] > 0 else None
    data = np.asarray(input_store.get_inline_allocation())
    kwargs: dict[str, object] = {
        "mode": "w",
        "shape": array_shape,
        "dtype": np.dtype(array_dtype),
        "chunks": chunks_arg,
    }
    if zarr_format == 3:
        kwargs["zarr_format"] = 3
    if not use_compressor:
        kwargs["compressor"] = None

    zarr.open_array(dirpath, **kwargs)[...] = data


def create_test_zarr(
    tmp_path: Path,
    data: np.ndarray,
    chunks: tuple[int, ...] | None = None,
    zarr_format: int = 2,
    use_compressor: bool = False,
) -> None:
    """Create a test Zarr array using a single-instance manual task.

    Parameters
    ----------
    tmp_path : Path
        Directory path for the zarr array.
    data : np.ndarray
        Data to write to the zarr array.
    chunks : tuple[int, ...] | None
        Chunk sizes, or None for default.
    zarr_format : int
        Zarr format version (2 or 3).
    use_compressor : bool
        If True, use zarr's default compressor. If False, no compression.
    """
    runtime = get_legate_runtime()

    store = runtime.create_store_from_buffer(
        Type.from_numpy_dtype(data.dtype), data.shape, data, read_only=True
    )

    if chunks is None:
        chunks_arg: tuple[int, ...] = (-1,)
    else:
        chunks_arg = tuple(chunks)

    # Create and execute a manual task with a single instance
    manual_task = runtime.create_manual_task(
        init_test_zarr_task.library, init_test_zarr_task.task_id, (1,)
    )

    manual_task.add_input(store)
    manual_task.add_scalar_arg(str(tmp_path), ty.string_type)
    manual_task.add_scalar_arg(data.shape, (ty.int64,))
    manual_task.add_scalar_arg(str(data.dtype), ty.string_type)
    manual_task.add_scalar_arg(chunks_arg, (ty.int64,))
    manual_task.add_scalar_arg(zarr_format, ty.int32)
    manual_task.add_scalar_arg(use_compressor, ty.bool_)
    manual_task.execute()
    runtime.issue_execution_fence(block=True)


def is_multi_gpu_ci() -> bool:
    from os import environ  # noqa: PLC0415

    from legate.core import TaskTarget  # noqa: PLC0415

    num_gpu = get_legate_runtime().get_machine().count(target=TaskTarget.GPU)
    return bool(num_gpu > 1 and "CI" in environ)


MULTI_GPU_CI = is_multi_gpu_ci()


@pytest.mark.xfail(
    int(zarr.__version__.split(".")[0]) >= 3,
    reason="tests do not support zarr v3",
)
class TestZarrV2:
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
        self,
        tmp_path: Path,
        shape: tuple[int, ...],
        chunks: tuple[int, ...],
        dtype: str,
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
    @pytest.mark.skipif(
        MULTI_GPU_CI,
        reason=(
            "Intermittent failures in CI for multi-gpu, "
            "see https://github.com/nv-legate/legate.internal/issues/2326"
        ),
    )
    def test_write_array_from_data_interface(
        self,
        tmp_path: Path,
        shape: tuple[int, ...],
        chunks: tuple[int, ...],
        dtype: str,
    ) -> None:
        a = np.arange(math.prod(shape), dtype=dtype).reshape(shape)
        store = get_legate_runtime().create_store_from_buffer(
            Type.from_numpy_dtype(a.dtype), a.shape, a, False
        )
        array = LogicalArray.from_store(store)
        table = Table.from_arrays(["store"], [array])

        write_array(ary=table, dirpath=tmp_path, chunks=chunks)
        get_legate_runtime().issue_execution_fence(block=True)

        b = zarr.open_array(tmp_path, mode="r")
        assert_array_equal(a, b)

    @pytest.mark.parametrize(*shape_chunks)
    @pytest.mark.parametrize("dtype", ["u1", "u8", "f8"])
    def test_read_array(
        self,
        tmp_path: Path,
        shape: tuple[int, ...],
        chunks: tuple[int, ...],
        dtype: str,
    ) -> None:
        """Test read of a Zarr array."""
        a = np.arange(math.prod(shape), dtype=dtype).reshape(shape)

        # Use helper function to create zarr array with single-instance task
        create_test_zarr(tmp_path, a, chunks=chunks)

        array = read_array(dirpath=tmp_path)
        b = np.asarray(array.get_physical_array())
        assert_array_equal(a, b)

    def test_read_compressor(self, tmp_path: Path) -> None:
        """Test read of a Zarr array with compressor."""
        a = np.array((1, 2, 3))

        # Use helper function to create zarr array with single-instance task
        # use_compressor=True to use zarr's default compressor
        create_test_zarr(tmp_path, a, use_compressor=True)

        msg = "compressor isn't supported"
        with pytest.raises(NotImplementedError, match=msg):
            read_array(dirpath=tmp_path)


@pytest.mark.skipif(
    int(zarr.__version__.split(".")[0]) < 3, reason="zarr v3 only tests"
)
class TestZarrV3:
    def test_read_array_v3(self, tmp_path: Path) -> None:
        """Test read of a v3 format Zarr array."""
        a = np.array((1, 2, 3))

        # Use helper function to create zarr array with single-instance task
        create_test_zarr(tmp_path, a, zarr_format=3)

        msg = "Zarr v3 support is not implemented yet"
        with pytest.raises(NotImplementedError, match=msg):
            read_array(dirpath=tmp_path)


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
