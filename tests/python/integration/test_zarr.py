# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import os
import sys
import json
import math
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, cast

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
from legate.core.experimental.io import zarr as legate_zarr
from legate.core.experimental.io.zarr import read_array, write_array
from legate.core.task import InputStore, task

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
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

ZARR_V2_DISK_FORMAT = 2
ZARR_V3_DISK_FORMAT = 3
ZARR_PYTHON_V3_MAJOR = 3
DEFAULT_CHUNK_HEURISTIC_CASES = (
    ((0,), "f8", (1,)),
    ((1000000000,), "u1", (1953125,)),
    ((1000000000,), "f8", (488282,)),
    ((2048, 257, 17), "f8", (512, 65, 5)),
)


def _zarr_major_version() -> int:
    return int(zarr.__version__.split(".", maxsplit=1)[0])


_ZARR_USES_SYNC_WORKERS = _zarr_major_version() >= ZARR_PYTHON_V3_MAJOR
_ZARR_SYNC_CALL_LOCK = threading.RLock()
_zarr_sync_cleanup: Callable[[], None] | None

# Production no longer calls zarr-python. These test helpers still do from
# Legate tasks and the main test thread, and zarr 3 sync calls need explicit
# cleanup around each sync API use.
if _ZARR_USES_SYNC_WORKERS:
    try:
        from zarr.core.sync import (  # type: ignore # noqa: PGH003
            cleanup_resources as _zarr_sync_cleanup,
        )
    except ImportError as ex:
        msg = "zarr-python 3 sync cleanup API moved"
        raise RuntimeError(msg) from ex
else:
    _zarr_sync_cleanup = None


def _cleanup_zarr_sync_resources() -> None:
    if _zarr_sync_cleanup is not None:
        _zarr_sync_cleanup()


@contextmanager
def _zarr_test_sync_call() -> Iterator[None]:
    if not _ZARR_USES_SYNC_WORKERS:
        yield
        return

    with _ZARR_SYNC_CALL_LOCK:
        gc_was_enabled = gc.isenabled()
        if gc_was_enabled:
            gc.disable()
        try:
            yield
        finally:
            try:
                _cleanup_zarr_sync_resources()
            finally:
                if gc_was_enabled:
                    gc.enable()


def _zarr_format_kwargs(zarr_format: int) -> dict[str, int]:
    if _zarr_major_version() >= ZARR_PYTHON_V3_MAJOR:
        return {"zarr_format": zarr_format}
    if zarr_format != ZARR_V2_DISK_FORMAT:
        return {"zarr_version": zarr_format}
    return {}


def _get_zarr_format(zarr_ary: Any) -> int:
    if hasattr(zarr_ary, "metadata"):
        return zarr_ary.metadata.zarr_format
    return ZARR_V2_DISK_FORMAT


def _read_zarr_chunks(dirpath: Path) -> tuple[int, ...]:
    with (dirpath / ".zarray").open(encoding="utf-8") as f:
        metadata = json.load(f)
    return tuple(int(item) for item in metadata["chunks"])


def _read_zarr_array(dirpath: Path) -> tuple[int, np.ndarray]:
    with _zarr_test_sync_call():
        zarr_ary = zarr.open_array(dirpath, mode="r")
        zarr_format = _get_zarr_format(zarr_ary)
        data = np.asarray(zarr_ary[:])
    return zarr_format, data


def _test_zarr_v2_metadata(**updates: object) -> dict[str, object]:
    metadata: dict[str, object] = {
        "chunks": [3],
        "compressor": None,
        "dimension_separator": ".",
        "dtype": "<i8",
        "fill_value": 0,
        "filters": None,
        "order": "C",
        "shape": [3],
        "zarr_format": ZARR_V2_DISK_FORMAT,
    }
    metadata.update(updates)
    return metadata


def _write_test_zarr_v2_metadata(
    dirpath: Path, metadata: dict[str, object]
) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)
    tmp_path = dirpath / f".zarray.{os.getpid()}.tmp"
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=True, indent=4, sort_keys=True)
        f.write("\n")
    tmp_path.replace(dirpath / ".zarray")


@task(variants=(VariantCode.CPU,))
def init_test_zarr_task(
    input_store: InputStore,
    dirpath: str,
    array_shape: tuple[int, ...],
    array_dtype: str,
    chunks: tuple[int, ...],
    use_default_chunks: bool,
    zarr_format: int,
    use_compressor: bool,
) -> None:
    """Task to initialize test Zarr array (single-instance, CPU-only).

    This task creates a Zarr array and populates it with data from the
    input store. It must run as a single instance to avoid race conditions.
    """
    chunks_arg = None if use_default_chunks else tuple(int(c) for c in chunks)
    data = np.asarray(input_store.get_inline_allocation())
    kwargs: dict[str, object] = {
        "mode": "w",
        "shape": tuple(int(s) for s in array_shape),
        "dtype": np.dtype(array_dtype),
        "chunks": chunks_arg,
    }
    kwargs.update(_zarr_format_kwargs(zarr_format))
    if not use_compressor and zarr_format < ZARR_V3_DISK_FORMAT:
        # This helper still emits v2 metadata when requested, so compressor
        # remains the v2 spelling even under zarr-python 3.
        kwargs["compressor"] = None

    with _zarr_test_sync_call():
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
        If True, use zarr's default compressor for v2. If False, create
        uncompressed v2 arrays. Ignored for v3, where this helper only creates
        enough metadata to exercise the format guard.
    """
    runtime = get_legate_runtime()

    store = runtime.create_store_from_buffer(
        Type.from_numpy_dtype(data.dtype), data.shape, data, read_only=True
    )

    use_default_chunks = chunks is None
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
    manual_task.add_scalar_arg(use_default_chunks, ty.bool_)
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

        _, b = _read_zarr_array(tmp_path)
        assert_array_equal(a, b)

    def test_write_scalar_array(self, tmp_path: Path) -> None:
        """Test write of a 0-dimensional Zarr array."""
        array = get_legate_runtime().create_array(ty.float64, ())

        with pytest.raises(
            ValueError,
            match="Cannot write a 0-dimensional \\(scalar\\) array to Zarr",
        ):
            write_array(ary=array, dirpath=tmp_path)

    def test_write_array_default_chunks_fixed_expectations(
        self, tmp_path: Path
    ) -> None:
        """Test chunks=None keeps the frozen v2 default chunking."""
        # Expected chunks were captured from zarr-python 2.18.7 and 3.2.1
        # using zarr.open_array(..., chunks=None, compressor=None) for v2
        # arrays. Keep the literals fixed so this test guards our owned copy of
        # the heuristic rather than tracking future zarr-python changes.
        actual_path = tmp_path / "actual"
        shape = (10,)
        dtype = "f8"
        expected_chunks = (10,)
        a = np.arange(math.prod(shape), dtype=dtype).reshape(shape)
        store = get_legate_runtime().create_store_from_buffer(
            Type.from_numpy_dtype(a.dtype), a.shape, a, False
        )
        array = LogicalArray.from_store(store)

        write_array(ary=array, dirpath=actual_path, chunks=None)
        get_legate_runtime().issue_execution_fence(block=True)

        assert _read_zarr_chunks(actual_path) == expected_chunks
        _, b = _read_zarr_array(actual_path)
        assert_array_equal(a, b)

    def test_write_array_minus_one_chunks_are_not_default(
        self, tmp_path: Path
    ) -> None:
        """Test explicit -1 chunks remain distinct from default chunks."""
        shape = (1_000_000,)
        dtype = "u1"
        expected_default_chunks = legate_zarr._guess_zarr_v2_chunks(
            shape, np.dtype(dtype).itemsize
        )
        assert expected_default_chunks != shape

        a = np.zeros(shape, dtype=dtype)
        store = get_legate_runtime().create_store_from_buffer(
            Type.from_numpy_dtype(a.dtype), a.shape, a, False
        )
        array = LogicalArray.from_store(store)

        default_path = tmp_path / "default"
        write_array(ary=array, dirpath=default_path, chunks=None)
        get_legate_runtime().issue_execution_fence(block=True)
        assert _read_zarr_chunks(default_path) == expected_default_chunks

        full_extent_path = tmp_path / "full-extent"
        write_array(ary=array, dirpath=full_extent_path, chunks=(-1,))
        get_legate_runtime().issue_execution_fence(block=True)
        assert _read_zarr_chunks(full_extent_path) == shape

    @pytest.mark.parametrize(
        ("shape", "dtype", "expected_chunks"), DEFAULT_CHUNK_HEURISTIC_CASES
    )
    def test_default_chunk_heuristic_large_fixed_expectations(
        self,
        shape: tuple[int, ...],
        dtype: str,
        expected_chunks: tuple[int, ...],
    ) -> None:
        """Test large-array branches without allocating the arrays."""
        actual = legate_zarr._guess_zarr_v2_chunks(
            shape, np.dtype(dtype).itemsize
        )
        assert actual == expected_chunks

    @pytest.mark.parametrize(
        ("chunks", "exc_type", "match"),
        [
            (False, TypeError, "chunks must be"),
            (True, TypeError, "chunks must be"),
            (0, ValueError, "chunks must be positive"),
            (-2, ValueError, "chunks must be positive"),
            ((1.5,), TypeError, "chunks must be"),
            ((np.float64(2),), TypeError, "chunks must be"),
            (("2",), TypeError, "chunks must be"),
            ((2, 0), ValueError, "chunks must be positive"),
        ],
    )
    def test_write_array_invalid_chunks_rejected(
        self,
        tmp_path: Path,
        chunks: object,
        exc_type: type[Exception],
        match: str,
    ) -> None:
        """Test invalid chunks are rejected before launching the write task."""
        shape = (10, 10)
        a = np.arange(math.prod(shape), dtype="f8").reshape(shape)
        store = get_legate_runtime().create_store_from_buffer(
            Type.from_numpy_dtype(a.dtype), a.shape, a, False
        )
        array = LogicalArray.from_store(store)

        with pytest.raises(exc_type, match=match):
            write_array(ary=array, dirpath=tmp_path, chunks=cast(Any, chunks))

    def test_write_array_unsupported_dtype(self, tmp_path: Path) -> None:
        """Test unsupported non-numeric dtypes fail before tile writes."""
        a = np.array([b"a", b"bb"], dtype="S4")
        store = get_legate_runtime().create_store_from_buffer(
            Type.from_numpy_dtype(a.dtype), a.shape, a, False
        )
        array = LogicalArray.from_store(store)

        with pytest.raises(NotImplementedError, match="numeric and bool"):
            write_array(ary=array, dirpath=tmp_path, chunks=(2,))

    def test_write_array_overwrites_existing_array(
        self, tmp_path: Path
    ) -> None:
        """Test write_array preserves zarr mode='w' overwrite behavior."""
        stale_file = tmp_path / "stale"
        a = np.arange(4, dtype="f8").reshape((2, 2))
        first = get_legate_runtime().create_store_from_buffer(
            Type.from_numpy_dtype(a.dtype), a.shape, a, False
        )
        first_array = LogicalArray.from_store(first)
        write_array(ary=first_array, dirpath=tmp_path, chunks=(2, 2))
        get_legate_runtime().issue_execution_fence(block=True)
        stale_file.write_text("stale", encoding="utf-8")

        b = np.arange(6, dtype="u1").reshape((3, 2))
        second = get_legate_runtime().create_store_from_buffer(
            Type.from_numpy_dtype(b.dtype), b.shape, b, False
        )
        second_array = LogicalArray.from_store(second)
        write_array(ary=second_array, dirpath=tmp_path, chunks=(3, 2))
        get_legate_runtime().issue_execution_fence(block=True)

        assert not stale_file.exists()
        _, c = _read_zarr_array(tmp_path)
        assert_array_equal(b, c)

    def test_write_array_refuses_non_zarr_directory(
        self, tmp_path: Path
    ) -> None:
        """Test write_array does not delete arbitrary non-Zarr directories."""
        existing = tmp_path / "not-zarr.txt"
        existing.write_text("keep me", encoding="utf-8")
        a = np.arange(4, dtype="f8")
        store = get_legate_runtime().create_store_from_buffer(
            Type.from_numpy_dtype(a.dtype), a.shape, a, False
        )
        array = LogicalArray.from_store(store)

        with pytest.raises(FileExistsError, match="recognized Zarr array"):
            write_array(ary=array, dirpath=tmp_path, chunks=(2,))

        assert existing.read_text(encoding="utf-8") == "keep me"

    @pytest.mark.parametrize("shape", [(10,), (4, 4)])
    def test_write_array_int_chunks(
        self, tmp_path: Path, shape: tuple[int, ...]
    ) -> None:
        """Test write of a Zarr array with integer chunks."""
        a = np.arange(math.prod(shape), dtype="f8").reshape(shape)
        store = get_legate_runtime().create_store_from_buffer(
            Type.from_numpy_dtype(a.dtype), a.shape, a, False
        )
        array = LogicalArray.from_store(store)

        write_array(ary=array, dirpath=tmp_path, chunks=2)
        get_legate_runtime().issue_execution_fence(block=True)

        _, b = _read_zarr_array(tmp_path)
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

        _, b = _read_zarr_array(tmp_path)
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
        _write_test_zarr_v2_metadata(
            tmp_path, _test_zarr_v2_metadata(compressor={"id": "zlib"})
        )

        msg = "compressor and filters aren't supported"
        with pytest.raises(NotImplementedError, match=msg):
            read_array(dirpath=tmp_path)

    def test_read_filters(self, tmp_path: Path) -> None:
        """Test read of a Zarr array with filters."""
        _write_test_zarr_v2_metadata(
            tmp_path,
            _test_zarr_v2_metadata(filters=[{"id": "delta", "dtype": "<i8"}]),
        )

        msg = "compressor and filters aren't supported"
        with pytest.raises(NotImplementedError, match=msg):
            read_array(dirpath=tmp_path)

    @pytest.mark.parametrize(
        ("key", "value", "match"),
        [
            ("order", "F", "C order"),
            ("dimension_separator", "/", "dimension_separator"),
        ],
    )
    def test_read_unsupported_v2_metadata(
        self, tmp_path: Path, key: str, value: str, match: str
    ) -> None:
        """Test unsupported v2 layout metadata is rejected."""
        _write_test_zarr_v2_metadata(
            tmp_path, _test_zarr_v2_metadata(**{key: value})
        )

        with pytest.raises(NotImplementedError, match=match):
            read_array(dirpath=tmp_path)

    @pytest.mark.parametrize(
        ("value", "exc_type", "match"),
        [
            (False, TypeError, "zarr_format.*integer"),
            ("2", TypeError, "zarr_format.*integer"),
            (1, NotImplementedError, "Only Zarr format 2"),
            (ZARR_V3_DISK_FORMAT, NotImplementedError, "format 3"),
        ],
    )
    def test_read_invalid_zarr_format_metadata(
        self,
        tmp_path: Path,
        value: object,
        exc_type: type[Exception],
        match: str,
    ) -> None:
        """Test unsupported and malformed format metadata is rejected."""
        _write_test_zarr_v2_metadata(
            tmp_path, _test_zarr_v2_metadata(zarr_format=value)
        )

        with pytest.raises(exc_type, match=match):
            read_array(dirpath=tmp_path)

    @pytest.mark.parametrize(
        "key", ["zarr_format", "shape", "chunks", "dtype"]
    )
    def test_read_missing_required_metadata_key(
        self, tmp_path: Path, key: str
    ) -> None:
        """Test malformed external metadata gets a clear error."""
        metadata = _test_zarr_v2_metadata()
        del metadata[key]
        _write_test_zarr_v2_metadata(tmp_path, metadata)

        with pytest.raises(ValueError, match=f"required key {key!r}"):
            read_array(dirpath=tmp_path)

    @pytest.mark.parametrize(
        ("key", "value", "exc_type", "match"),
        [
            ("shape", [1.5], TypeError, "items.*integers"),
            ("shape", [True], TypeError, "items.*integers"),
            ("shape", ["1"], TypeError, "items.*integers"),
            ("shape", [-1], ValueError, "non-negative"),
            ("shape", [3, 3], ValueError, "same rank"),
            ("chunks", [1.5], TypeError, "items.*integers"),
            ("chunks", [False], TypeError, "items.*integers"),
            ("chunks", ["1"], TypeError, "items.*integers"),
            ("chunks", [0], ValueError, "positive integers"),
            ("chunks", [-1], ValueError, "positive integers"),
            ("chunks", [3, 3], ValueError, "same rank"),
        ],
    )
    def test_read_invalid_shape_or_chunks_metadata(
        self,
        tmp_path: Path,
        key: str,
        value: list[object],
        exc_type: type[Exception],
        match: str,
    ) -> None:
        """Test malformed external shape/chunk metadata is rejected early."""
        _write_test_zarr_v2_metadata(
            tmp_path, _test_zarr_v2_metadata(**{key: value})
        )

        with pytest.raises(exc_type, match=match):
            read_array(dirpath=tmp_path)


@pytest.mark.skipif(
    _zarr_major_version() < ZARR_PYTHON_V3_MAJOR, reason="zarr v3 only tests"
)
class TestZarrV3:
    def test_write_array_v2_format(self, tmp_path: Path) -> None:
        """Test that write_array writes v2-format arrays under zarr v3."""
        a = np.arange(6, dtype="f8").reshape((2, 3))
        store = get_legate_runtime().create_store_from_buffer(
            Type.from_numpy_dtype(a.dtype), a.shape, a, False
        )
        array = LogicalArray.from_store(store)

        write_array(ary=array, dirpath=tmp_path, chunks=(2, 3))
        get_legate_runtime().issue_execution_fence(block=True)

        zarr_format, b = _read_zarr_array(tmp_path)
        assert zarr_format == ZARR_V2_DISK_FORMAT
        assert_array_equal(a, b)

    def test_read_array_v3(self, tmp_path: Path) -> None:
        """Test read of a v3 format Zarr array."""
        a = np.array((1, 2, 3))

        # Use helper function to create zarr array with single-instance task
        create_test_zarr(tmp_path, a, zarr_format=3)

        msg = "Zarr format 3 support is not implemented yet"
        with pytest.raises(NotImplementedError, match=msg):
            read_array(dirpath=tmp_path)


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
