# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import gc
import weakref

import numpy as np

import pytest

from legate.core import (
    DimOrdering,
    ExternalAllocation,
    TaskTarget,
    get_legate_runtime,
    types as ty,
)

try:
    import cupy  # type: ignore[import-not-found]
except ModuleNotFoundError:
    cupy = None

try:
    import torch  # type: ignore[import-not-found]
except ModuleNotFoundError:
    torch = None


def test_sysmem_single_tile() -> None:
    runtime = get_legate_runtime()

    buf = np.arange(10, dtype=np.float64)
    alloc = ExternalAllocation.from_sysmem(
        buf.ctypes.data, buf.nbytes, read_only=True, source=buf
    )

    store, _ = runtime.create_store_from_tiles(
        dtype=ty.float64,
        shape=(10,),
        tile_shape=(10,),
        allocations=[(alloc, (0,))],
    )

    result = np.from_dlpack(store.get_physical_store())
    expected = np.arange(10, dtype=np.float64)
    np.testing.assert_allclose(result, expected)
    store.detach()


def test_from_dlpack_single_tile() -> None:
    runtime = get_legate_runtime()

    buf = np.arange(10, dtype=np.float64)
    alloc = ExternalAllocation.from_dlpack(buf, read_only=True)

    store, _ = runtime.create_store_from_tiles(
        dtype=ty.float64,
        shape=(10,),
        tile_shape=(10,),
        allocations=[(alloc, (0,))],
    )

    result = np.from_dlpack(store.get_physical_store())
    expected = np.arange(10, dtype=np.float64)
    np.testing.assert_allclose(result, expected)
    store.detach()


def test_sysmem_multi_tile() -> None:
    runtime = get_legate_runtime()
    val1, val2 = 42, 84

    buf1 = np.full(10, val1, dtype=np.int64)
    buf2 = np.full(10, val2, dtype=np.int64)

    alloc1 = ExternalAllocation.from_sysmem(
        buf1.ctypes.data, buf1.nbytes, read_only=True, source=buf1
    )
    alloc2 = ExternalAllocation.from_sysmem(
        buf2.ctypes.data, buf2.nbytes, read_only=True, source=buf2
    )

    store, _ = runtime.create_store_from_tiles(
        dtype=ty.int64,
        shape=(20,),
        tile_shape=(10,),
        allocations=[(alloc1, (0,)), (alloc2, (1,))],
    )

    result = np.from_dlpack(store.get_physical_store())
    expected = np.concatenate(
        [np.full(10, val1, dtype=np.int64), np.full(10, val2, dtype=np.int64)]
    )
    np.testing.assert_array_equal(result, expected)
    store.detach()


def test_from_dlpack_multi_tile() -> None:
    runtime = get_legate_runtime()

    tile_a = np.arange(10, dtype=np.float64)
    tile_b = np.arange(10, 20, dtype=np.float64)

    alloc_a = ExternalAllocation.from_dlpack(tile_a, read_only=True)
    alloc_b = ExternalAllocation.from_dlpack(tile_b, read_only=True)

    store, _ = runtime.create_store_from_tiles(
        dtype=ty.float64,
        shape=(20,),
        tile_shape=(10,),
        allocations=[(alloc_a, (0,)), (alloc_b, (1,))],
    )

    result = np.from_dlpack(store.get_physical_store())
    expected = np.arange(20, dtype=np.float64)
    np.testing.assert_allclose(result, expected)
    store.detach()


def test_create_store_from_tiles_with_explicit_ordering() -> None:
    runtime = get_legate_runtime()

    buf = np.arange(10, dtype=np.float64)
    alloc = ExternalAllocation.from_sysmem(
        buf.ctypes.data, buf.nbytes, read_only=True, source=buf
    )

    store, _ = runtime.create_store_from_tiles(
        dtype=ty.float64,
        shape=(10,),
        tile_shape=(10,),
        allocations=[(alloc, (0,))],
        ordering=DimOrdering.c_order(),
    )

    result = np.from_dlpack(store.get_physical_store())
    np.testing.assert_allclose(result, buf)
    store.detach()


def test_create_store_from_tiles_rejects_bad_ordering() -> None:
    runtime = get_legate_runtime()

    buf = np.arange(10, dtype=np.float64)
    alloc = ExternalAllocation.from_sysmem(
        buf.ctypes.data, buf.nbytes, read_only=True, source=buf
    )

    with pytest.raises(
        TypeError, match="ordering must be a DimOrdering instance"
    ):
        runtime.create_store_from_tiles(
            dtype=ty.float64,
            shape=(10,),
            tile_shape=(10,),
            allocations=[(alloc, (0,))],
            ordering=object(),  # type: ignore[arg-type]
        )


def test_sum_over_tiles() -> None:
    runtime = get_legate_runtime()
    node_id = runtime.node_id
    node_count = runtime.node_count

    val1 = float(node_id * 2 + 1)
    val2 = float(node_id * 2 + 2)

    buf1 = np.full(10, val1, dtype=np.float64)
    buf2 = np.full(10, val2, dtype=np.float64)

    alloc1 = ExternalAllocation.from_sysmem(
        buf1.ctypes.data, buf1.nbytes, read_only=True, source=buf1
    )
    alloc2 = ExternalAllocation.from_sysmem(
        buf2.ctypes.data, buf2.nbytes, read_only=True, source=buf2
    )

    global_size = 20 * node_count
    store, _ = runtime.create_store_from_tiles(
        dtype=ty.float64,
        shape=(global_size,),
        tile_shape=(10,),
        allocations=[(alloc1, (node_id * 2,)), (alloc2, (node_id * 2 + 1,))],
    )

    result = np.from_dlpack(store.get_physical_store())
    total = float(result.sum())

    expected = 0.0
    for n in range(node_count):
        expected += 10 * float(n * 2 + 1) + 10 * float(n * 2 + 2)

    np.testing.assert_allclose(total, expected)
    store.detach()


def test_detach_lifecycle() -> None:
    runtime = get_legate_runtime()

    buf = np.arange(10, dtype=np.float64)
    buf_copy = buf.copy()

    alloc = ExternalAllocation.from_sysmem(
        buf.ctypes.data, buf.nbytes, read_only=True, source=buf
    )

    store, _ = runtime.create_store_from_tiles(
        dtype=ty.float64,
        shape=(10,),
        tile_shape=(10,),
        allocations=[(alloc, (0,))],
    )

    result = np.from_dlpack(store.get_physical_store())
    np.testing.assert_allclose(result, buf_copy)

    store.detach()

    np.testing.assert_array_equal(buf, buf_copy)


def test_source_outlives_alloc_wrapper() -> None:
    runtime = get_legate_runtime()

    buf = np.arange(10, dtype=np.float64)
    buf_copy = buf.copy()
    weak_buf = weakref.ref(buf)

    alloc = ExternalAllocation.from_sysmem(
        buf.ctypes.data, buf.nbytes, read_only=True, source=buf
    )
    store, _ = runtime.create_store_from_tiles(
        dtype=ty.float64,
        shape=(10,),
        tile_shape=(10,),
        allocations=[(alloc, (0,))],
    )

    del alloc
    del buf
    gc.collect()
    assert weak_buf() is not None

    result = np.from_dlpack(store.get_physical_store())
    np.testing.assert_allclose(result, buf_copy)

    store.detach()
    gc.collect()
    assert weak_buf() is None


def test_not_read_only_rejected() -> None:
    runtime = get_legate_runtime()
    buf = np.zeros(10, dtype=np.int64)

    alloc = ExternalAllocation.from_sysmem(
        buf.ctypes.data, buf.nbytes, read_only=False, source=buf
    )

    with pytest.raises((ValueError, RuntimeError)):
        runtime.create_store_from_tiles(
            dtype=ty.int64,
            shape=(10,),
            tile_shape=(10,),
            allocations=[(alloc, (0,))],
        )


def test_alloc_usable_after_throw() -> None:
    runtime = get_legate_runtime()
    buf1 = np.zeros(10, dtype=np.int64)
    buf2 = np.zeros(10, dtype=np.int64)

    alloc1 = ExternalAllocation.from_sysmem(
        buf1.ctypes.data, buf1.nbytes, read_only=True, source=buf1
    )
    alloc2 = ExternalAllocation.from_sysmem(
        buf2.ctypes.data, buf2.nbytes, read_only=False, source=buf2
    )

    with pytest.raises(ValueError, match="must be read-only"):
        runtime.create_store_from_tiles(
            dtype=ty.int64,
            shape=(20,),
            tile_shape=(10,),
            allocations=[(alloc1, (0,)), (alloc2, (1,))],
        )

    assert alloc1.size == buf1.nbytes
    assert alloc2.size == buf2.nbytes
    assert alloc1.read_only is True
    assert alloc2.read_only is False


def test_buffer_too_small() -> None:
    runtime = get_legate_runtime()
    buf = np.zeros(9, dtype=np.int64)

    alloc = ExternalAllocation.from_sysmem(
        buf.ctypes.data, buf.nbytes, read_only=True, source=buf
    )

    with pytest.raises((ValueError, RuntimeError)):
        runtime.create_store_from_tiles(
            dtype=ty.int64,
            shape=(10,),
            tile_shape=(10,),
            allocations=[(alloc, (0,))],
        )


def test_duplicate_color() -> None:
    runtime = get_legate_runtime()
    buf = np.zeros(10, dtype=np.int64)

    alloc = ExternalAllocation.from_sysmem(
        buf.ctypes.data, buf.nbytes, read_only=True, source=buf
    )

    with pytest.raises((ValueError, RuntimeError)):
        runtime.create_store_from_tiles(
            dtype=ty.int64,
            shape=(10,),
            tile_shape=(10,),
            allocations=[(alloc, (0,)), (alloc, (0,))],
        )


def test_wrong_allocation_type() -> None:
    runtime = get_legate_runtime()

    with pytest.raises(TypeError):
        runtime.create_store_from_tiles(
            dtype=ty.int64,
            shape=(10,),
            tile_shape=(10,),
            allocations=[("not_an_allocation", (0,))],  # type: ignore[list-item]
        )


@pytest.mark.skipif(
    len(get_legate_runtime().machine.only(TaskTarget.GPU)) == 0,
    reason="not severe: test requires GPU",
)
def test_from_dlpack_cupy_single_tile() -> None:
    if cupy is None:
        pytest.skip("not severe: cupy not available")

    runtime = get_legate_runtime()

    buf = cupy.arange(10, dtype=cupy.float64)
    alloc = ExternalAllocation.from_dlpack(buf, read_only=True)

    store, _ = runtime.create_store_from_tiles(
        dtype=ty.float64,
        shape=(10,),
        tile_shape=(10,),
        allocations=[(alloc, (0,))],
    )

    result = np.from_dlpack(store.get_physical_store())
    expected = np.arange(10, dtype=np.float64)
    np.testing.assert_allclose(result, expected)
    store.detach()


@pytest.mark.skipif(
    len(get_legate_runtime().machine.only(TaskTarget.GPU)) == 0,
    reason="not severe: test requires GPU",
)
def test_from_dlpack_torch_gpu() -> None:
    if torch is None:
        pytest.skip("not severe: torch not available")
    if not torch.cuda.is_available():
        pytest.skip("not severe: CUDA not available")

    runtime = get_legate_runtime()

    buf = torch.arange(10, dtype=torch.float64, device="cuda:0")
    alloc = ExternalAllocation.from_dlpack(buf, read_only=True)

    store, _ = runtime.create_store_from_tiles(
        dtype=ty.float64,
        shape=(10,),
        tile_shape=(10,),
        allocations=[(alloc, (0,))],
    )

    result = np.from_dlpack(store.get_physical_store())
    expected = np.arange(10, dtype=np.float64)
    np.testing.assert_allclose(result, expected)
    store.detach()


@pytest.mark.skipif(
    len(get_legate_runtime().machine.only(TaskTarget.GPU)) == 0,
    reason="not severe: test requires GPU",
)
def test_from_dlpack_torch_multi_tile() -> None:
    if torch is None:
        pytest.skip("not severe: torch not available")
    if not torch.cuda.is_available():
        pytest.skip("not severe: CUDA not available")

    runtime = get_legate_runtime()

    tile_a = torch.arange(10, dtype=torch.float64, device="cuda:0")
    tile_b = torch.arange(10, 20, dtype=torch.float64, device="cuda:0")

    alloc_a = ExternalAllocation.from_dlpack(tile_a, read_only=True)
    alloc_b = ExternalAllocation.from_dlpack(tile_b, read_only=True)

    store, _ = runtime.create_store_from_tiles(
        dtype=ty.float64,
        shape=(20,),
        tile_shape=(10,),
        allocations=[(alloc_a, (0,)), (alloc_b, (1,))],
    )

    result = np.from_dlpack(store.get_physical_store())
    expected = np.arange(20, dtype=np.float64)
    np.testing.assert_allclose(result, expected)
    store.detach()


@pytest.mark.skipif(
    get_legate_runtime()
    .machine.get_processor_range(TaskTarget.GPU)
    .per_node_count
    < 2,
    reason="not severe: test requires >= 2 GPUs per rank",
)
def test_multi_gpu_one_tile_per_gpu() -> None:
    if cupy is None:
        pytest.skip("not severe: cupy not available")

    runtime = get_legate_runtime()
    n_gpus = runtime.machine.get_processor_range(TaskTarget.GPU).per_node_count
    allocs = []

    for gpu_id in range(n_gpus):
        with cupy.cuda.Device(gpu_id):
            buf = cupy.full(10, float(gpu_id + 1), dtype=cupy.float64)
            alloc = ExternalAllocation.from_dlpack(buf, read_only=True)
            allocs.append((alloc, (gpu_id,)))

    store, _ = runtime.create_store_from_tiles(
        dtype=ty.float64,
        shape=(10 * n_gpus,),
        tile_shape=(10,),
        allocations=allocs,  # type: ignore[arg-type]
    )

    result = np.from_dlpack(store.get_physical_store())
    for gpu_id in range(n_gpus):
        shard = result[gpu_id * 10 : (gpu_id + 1) * 10]
        expected_val = float(gpu_id + 1)
        np.testing.assert_allclose(shard, expected_val)
    store.detach()


@pytest.mark.skipif(
    get_legate_runtime()
    .machine.get_processor_range(TaskTarget.GPU)
    .per_node_count
    < 2,
    reason="not severe: test requires >= 2 GPUs per rank",
)
def test_multi_gpu_sum() -> None:
    if cupy is None:
        pytest.skip("not severe: cupy not available")

    runtime = get_legate_runtime()
    n_gpus = runtime.machine.get_processor_range(TaskTarget.GPU).per_node_count
    allocs = []

    for gpu_id in range(n_gpus):
        with cupy.cuda.Device(gpu_id):
            buf = cupy.full(10, float(gpu_id + 1), dtype=cupy.float64)
            alloc = ExternalAllocation.from_dlpack(buf, read_only=True)
            allocs.append((alloc, (gpu_id,)))

    store, _ = runtime.create_store_from_tiles(
        dtype=ty.float64,
        shape=(10 * n_gpus,),
        tile_shape=(10,),
        allocations=allocs,  # type: ignore[arg-type]
    )

    result = np.from_dlpack(store.get_physical_store())
    total = float(result.sum())
    expected = sum(10 * float(g + 1) for g in range(n_gpus))
    np.testing.assert_allclose(total, expected)
    store.detach()


@pytest.mark.skipif(
    get_legate_runtime().node_count < 2,
    reason="not severe: test requires >= 2 ranks",
)
def test_multi_rank_one_tile_per_rank() -> None:
    runtime = get_legate_runtime()
    node_id = runtime.node_id
    node_count = runtime.node_count

    val = float(node_id)
    buf = np.full(10, val, dtype=np.float64)

    alloc = ExternalAllocation.from_sysmem(
        buf.ctypes.data, buf.nbytes, read_only=True, source=buf
    )

    store, _ = runtime.create_store_from_tiles(
        dtype=ty.float64,
        shape=(10 * node_count,),
        tile_shape=(10,),
        allocations=[(alloc, (node_id,))],
    )

    result = np.from_dlpack(store.get_physical_store())
    total = float(result.sum())
    expected = 10 * sum(float(n) for n in range(node_count))
    np.testing.assert_allclose(total, expected)
    store.detach()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
