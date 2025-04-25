# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import pytest

from legate.core import (
    StoreTarget,
    TaskTarget,
    Type,
    VariantCode,
    get_legate_runtime,
    types as ty,
)
from legate.core.task import InputStore, task


def compute_strides(shape: tuple[int, ...], dtype: Type) -> tuple[int, ...]:
    if not len(shape):
        return ()

    # Strides for an array with an itemsize of 1 in C-order
    tmp = shape[::-1]
    tmp = (1, *np.cumprod(tmp[:-1]))
    # Adjust for the real itemsize
    return tuple(dtype.size * v for v in reversed(tmp))


class TestInlineAllocation:
    @pytest.mark.parametrize("dtype", (ty.int8, ty.int16, ty.int32, ty.int64))
    @pytest.mark.parametrize("shape", ((1,), (1, 2, 3), (1, 2, 3, 4)))
    def test_basic(self, dtype: Type, shape: tuple[int, ...]) -> None:
        store = get_legate_runtime().create_store(dtype=dtype, shape=shape)
        store.fill(0)
        alloc = store.get_physical_store().get_inline_allocation()
        strides = compute_strides(shape, dtype)

        assert alloc.ptr != 0
        assert alloc.strides == strides
        assert alloc.shape == shape
        assert alloc.target == StoreTarget.SYSMEM
        assert alloc.__array_interface__ == {
            "version": 3,
            "shape": shape,
            "typestr": dtype.to_numpy_dtype().str,
            "data": (alloc.ptr, False),
            "strides": strides,
        }
        with pytest.raises(
            ValueError,
            match=(
                r"Physical store in a host-only memory does not support "
                "the CUDA array interface"
            ),
        ):
            _ = alloc.__cuda_array_interface__

    @pytest.mark.parametrize("dtype", (ty.int8, ty.int16, ty.int32, ty.int64))
    @pytest.mark.parametrize("shape", ((1,), (1, 2, 3), (1, 2, 3, 4)))
    def test_to_numpy_array(self, dtype: Type, shape: tuple[int, ...]) -> None:
        val = -33
        store = get_legate_runtime().create_store(dtype=dtype, shape=shape)
        store.fill(val)
        alloc = store.get_physical_store().get_inline_allocation()
        arr = np.asarray(alloc)

        assert alloc.shape == arr.shape
        assert alloc.strides == arr.strides
        assert dtype.to_numpy_dtype() == arr.dtype
        assert (arr == val).all()

    @pytest.mark.skipif(
        get_legate_runtime().get_machine().only(TaskTarget.GPU).empty,
        reason="This test requires GPUs",
    )
    @pytest.mark.parametrize("dtype", (ty.int8, ty.int16, ty.int32, ty.int64))
    @pytest.mark.parametrize("shape", ((1,), (1, 2, 3), (1, 2, 3, 4)))
    def test_to_cupy_array(self, dtype: Type, shape: tuple[int, ...]) -> None:
        import cupy as cp  # type: ignore[import-not-found]

        @task(variants=(VariantCode.GPU,))
        def foo(x: InputStore, val: int) -> None:
            alloc = x.get_inline_allocation()
            arr = cp.asarray(alloc)

            assert alloc.shape == arr.shape
            assert alloc.strides == arr.strides
            assert dtype.to_numpy_dtype() == arr.dtype
            assert (arr == val).all()

        val = 100
        store = get_legate_runtime().create_store(dtype=dtype, shape=shape)
        store.fill(val)

        foo(store, val)
        get_legate_runtime().issue_execution_fence(block=True)

    @pytest.mark.parametrize("dtype", (ty.int8, ty.int16, ty.int32, ty.int64))
    @pytest.mark.parametrize("shape", ((1,), (1, 2, 3), (1, 2, 3, 4)))
    def test_buffer_protocol(
        self, dtype: Type, shape: tuple[int, ...]
    ) -> None:
        val = 42
        store = get_legate_runtime().create_store(dtype=dtype, shape=shape)
        store.fill(val)
        alloc = store.get_physical_store().get_inline_allocation()
        strides = compute_strides(shape, dtype)

        mv = memoryview(alloc)

        assert mv.itemsize == dtype.size
        assert mv.ndim == len(shape)
        assert mv.shape == shape
        assert mv.strides == strides
        assert mv.contiguous

    @pytest.mark.skipif(
        get_legate_runtime().get_machine().only(TaskTarget.GPU).empty,
        reason="This test requires GPUs",
    )
    @pytest.mark.parametrize("dtype", (ty.int8, ty.int16, ty.int32, ty.int64))
    @pytest.mark.parametrize("shape", ((1,), (1, 2, 3), (1, 2, 3, 4)))
    def test_stream(self, dtype: Type, shape: tuple[int, ...]) -> None:
        strides = compute_strides(shape, dtype)

        @task(variants=(VariantCode.GPU,))
        def foo(x: InputStore) -> None:
            alloc = x.get_inline_allocation()

            assert alloc.ptr != 0
            assert alloc.strides == strides
            assert alloc.shape == shape
            assert alloc.target == StoreTarget.FBMEM
            with pytest.raises(
                ValueError,
                match=(
                    r"Physical store in a framebuffer memory does not support "
                    "the array interface"
                ),
            ):
                _ = alloc.__array_interface__
            # Cannot compare the dictionary directly, because we want to test
            # that the stream value is *not* some value
            cai = alloc.__cuda_array_interface__
            assert cai["version"] == 3
            assert cai["shape"] == shape
            assert cai["typestr"] == dtype.to_numpy_dtype().str
            assert cai["data"] == (alloc.ptr, False)
            assert cai["strides"] == strides
            # stream is either None, or an integer. If None, then no sync is
            # required by the consumer. 0 is disallowed outright because it is
            # ambiguous with None. 1 is the legacy default stream, and 2 is the
            # per-thread default stream. Since stream is always set from the
            # task stream, it is never any of these.
            #
            # See
            # https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html#python-interface-specification
            assert "stream" in cai
            stream = cai["stream"]
            assert stream is not None
            assert isinstance(stream, int)
            assert stream not in {0, 1, 2}

        store = get_legate_runtime().create_store(dtype=dtype, shape=shape)
        store.fill(123)

        foo(store)
        get_legate_runtime().issue_execution_fence(block=True)
