# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import re

import pytest

from legate.core import (
    Scalar,
    StoreTarget,
    TaskTarget,
    VariantCode,
    get_legate_runtime,
    types as ty,
)
from legate.core.task import OutputStore, task


class Test_store_creation:
    def test_bound(self) -> None:
        runtime = get_legate_runtime()
        store = runtime.create_store(ty.int64, shape=(4, 4))
        assert not store.unbound
        assert store.ndim == 2
        assert store.shape == (4, 4)
        assert store.type == ty.int64
        assert not store.transformed
        assert store.extents == store.shape
        assert store.volume == 16
        assert store.size == store.volume
        assert repr(store) == str(store)
        # touching raw_handle for coverage
        _ = store.raw_handle

    def test_unbound(self) -> None:
        runtime = get_legate_runtime()
        store = runtime.create_store(ty.int64)
        assert store.unbound
        assert store.ndim == 1
        assert store.type == ty.int64
        assert not store.transformed
        with pytest.raises(ValueError):  # noqa: PT011
            _ = store.shape.extents

    def test_ndim(self) -> None:
        runtime = get_legate_runtime()
        store = runtime.create_store(ty.int64, ndim=2)
        assert store.unbound
        assert store.ndim == 2
        assert store.type == ty.int64
        assert not store.transformed

    def test_optimize_scalar(self) -> None:
        runtime = get_legate_runtime()
        store = runtime.create_store(
            ty.int64, shape=(1,), optimize_scalar=True
        )
        assert store.has_scalar_storage

    def test_create_store_from_scalar(self) -> None:
        runtime = get_legate_runtime()
        store = runtime.create_store_from_scalar(Scalar(123, ty.int64))
        assert not store.unbound
        assert store.ndim == 1
        assert store.shape == (1,)
        assert store.type == ty.int64
        assert not store.transformed

    @pytest.mark.parametrize("shape", [(1,), (1, 1), (1, 1, 1)], ids=str)
    def test_create_store_from_scalar_shape(self, shape: tuple[int]) -> None:
        runtime = get_legate_runtime()
        store = runtime.create_store_from_scalar(
            Scalar(123, ty.int64), shape=shape
        )
        assert not store.unbound
        assert store.ndim == len(shape)
        assert store.shape == shape
        assert store.type == ty.int64
        assert not store.transformed


class Test_store_creation_error:
    def test_ndim_with_shape(self) -> None:
        runtime = get_legate_runtime()
        with pytest.raises(ValueError, match="ndim cannot be used with shape"):
            runtime.create_store(ty.int32, shape=(1,), ndim=1)

    def test_scalar_size_mismatch(self) -> None:
        runtime = get_legate_runtime()
        msg = (
            "Type int32 expects a value of size 4, but the size of value is 5"
        )
        with pytest.raises(ValueError, match=msg):
            runtime.create_store_from_scalar(Scalar(b"12345", ty.int32))

    def test_scalar_volume_mismatch(self) -> None:
        runtime = get_legate_runtime()
        msg = "Scalar stores must have a shape of volume 1"
        with pytest.raises(ValueError, match=msg):
            runtime.create_store_from_scalar(Scalar(123, ty.int64), (1, 2))


class Test_store_valid_transform:
    def test_bound(self) -> None:
        runtime = get_legate_runtime()
        store = runtime.create_store(ty.int64, shape=(4, 3))

        promoted = store.promote(0, 5)
        assert promoted.shape == (5, 4, 3)
        assert promoted.transformed

        projected = store.project(0, 1)
        assert projected.shape == (3,)
        assert projected.transformed

        sliced = store.slice(1, slice(1, 3))
        assert sliced.shape == (4, 2)
        assert sliced.transformed

        transposed = store.transpose((1, 0))
        assert transposed.shape == (3, 4)
        assert transposed.transformed

        delinearized = store.delinearize(0, (2, 2))
        assert delinearized.shape == (2, 2, 3)
        assert delinearized.transformed


class Test_store_invalid_transform:
    def test_bound(self) -> None:
        runtime = get_legate_runtime()
        store = runtime.create_store(ty.int64, shape=(4, 3))

        with pytest.raises(ValueError):  # noqa: PT011
            store.promote(3, 5)

        with pytest.raises(ValueError):  # noqa: PT011
            store.promote(-3, 5)

        with pytest.raises(ValueError):  # noqa: PT011
            store.project(2, 1)

        with pytest.raises(ValueError):  # noqa: PT011
            store.project(-3, 1)

        with pytest.raises(ValueError):  # noqa: PT011
            store.project(0, 4)

        with pytest.raises(ValueError):  # noqa: PT011
            store.slice(2, slice(1, 3))

        with pytest.raises(NotImplementedError):
            store.slice(1, slice(1, 3, 2))

        with pytest.raises(ValueError):  # noqa: PT011
            store.slice(1, slice(1, 4))

        with pytest.raises(ValueError):  # noqa: PT011
            store.transpose((2,))

        with pytest.raises(ValueError):  # noqa: PT011
            store.transpose((0, 0))

        with pytest.raises(ValueError):  # noqa: PT011
            store.transpose((2, 0))

        msg = re.escape("Expected an iterable but got <class 'int'>")
        with pytest.raises(ValueError, match=msg):
            store.transpose(1)  # type: ignore[arg-type]

        with pytest.raises(ValueError):  # noqa: PT011
            store.delinearize(2, (2, 3))

        with pytest.raises(ValueError):  # noqa: PT011
            store.delinearize(0, (2, 3))

    def test_unbound(self) -> None:
        runtime = get_legate_runtime()
        store = runtime.create_store(ty.int64)
        with pytest.raises(ValueError):  # noqa: PT011
            store.promote(1, 1)


def get_num_gpus_() -> int:
    machine = get_legate_runtime().get_machine()
    return len(machine.only(TaskTarget.GPU))


class Test_offload_to:
    @pytest.mark.skipif(get_num_gpus_() == 0, reason="No GPUs found")
    def test_host_offload(self) -> None:
        runtime = get_legate_runtime()
        # TODO(amberhassaan): This test either needs access to the amount of
        # fbmem allotted to Legate runtime or it needs to be run with Legate
        # configured with a known amount of fbmem so that a big enough store
        # can be created that will necessitate offloading.
        one_meg = 1024 * 1024
        shape = (128 * one_meg,)
        # two stores too big for the GPU memory
        store1 = runtime.create_store(dtype=ty.int8, shape=shape)
        store2 = runtime.create_store(dtype=ty.int8, shape=shape)

        # launch two tasks that access the GPU memory
        @task(variants=(VariantCode.GPU,))
        def task_gpu(store: OutputStore) -> None:
            pass

        task_gpu(store1)
        store1.offload_to(StoreTarget.SYSMEM)
        task_gpu(store2)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
