# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys
from typing import Any

import numpy as np

import pytest

from legate import install_info
from legate.core import ExternalAllocation, TaskTarget, get_legate_runtime

try:
    import cupy  # type: ignore[import-not-found]

    cupy.cuda.runtime.getDeviceCount()
except Exception:
    cupy = None


class _UnversionedDLPack:
    def __init__(self, arr: np.ndarray[Any, Any]) -> None:
        self.arr = arr

    def __dlpack__(
        self,
        stream: int | object | None = None,
        dl_device: tuple[int, int] | None = None,
        copy: bool | None = None,
    ) -> object:
        return self.arr.__dlpack__(
            stream=stream, dl_device=dl_device, copy=copy
        )


class TestFromDLPack:
    def test_numpy_1d(self) -> None:
        buf = np.arange(10, dtype=np.float64)
        alloc = ExternalAllocation.from_dlpack(buf, read_only=True)
        assert alloc.size == buf.nbytes
        assert alloc.read_only is True

    @pytest.mark.parametrize(
        "dtype", (np.int32, np.float32, np.float64, np.complex128)
    )
    def test_numpy_dtypes(self, dtype: np.dtype) -> None:
        buf = np.ones(10, dtype=dtype)
        alloc = ExternalAllocation.from_dlpack(buf)
        assert alloc.size == buf.nbytes

    @pytest.mark.parametrize("shape", ((10,), (4, 4), (2, 3, 4)))
    def test_numpy_shapes(self, shape: tuple[int, ...]) -> None:
        buf = np.ones(shape, dtype=np.float64)
        alloc = ExternalAllocation.from_dlpack(buf)
        assert alloc.size == buf.nbytes

    def test_read_only_true(self) -> None:
        buf = np.zeros(10, dtype=np.float64)
        alloc = ExternalAllocation.from_dlpack(buf, read_only=True)
        assert alloc.read_only is True

    def test_read_only_false(self) -> None:
        buf = np.zeros(10, dtype=np.float64)
        alloc = ExternalAllocation.from_dlpack(buf, read_only=False)
        assert alloc.read_only is False

    def test_rejects_non_dlpack(self) -> None:
        with pytest.raises(AttributeError, match="__dlpack__"):
            ExternalAllocation.from_dlpack([1, 2, 3])

    def test_rejects_no_dlpack_method(self) -> None:
        class FakeDLPackDevice:
            def __dlpack_device__(self) -> tuple[int, int]:
                return (1, 0)

        with pytest.raises(AttributeError, match="__dlpack__"):
            ExternalAllocation.from_dlpack(FakeDLPackDevice())

    @pytest.mark.skipif(
        cupy is None
        or get_legate_runtime().get_machine().only(TaskTarget.GPU).empty,
        reason="not severe: test requires cupy and GPUs",
    )
    def test_cupy_gpu(self) -> None:
        buf = cupy.zeros(10, dtype=cupy.float64)
        alloc = ExternalAllocation.from_dlpack(buf, read_only=True)
        assert alloc.size == buf.nbytes
        assert alloc.read_only is True

    def test_source_ref_held(self) -> None:
        buf = np.zeros(10, dtype=np.float64)
        refcount_before = sys.getrefcount(buf)
        alloc = ExternalAllocation.from_dlpack(buf)
        refcount_after = sys.getrefcount(buf)
        assert refcount_after > refcount_before
        del alloc

    def test_rejects_non_contiguous_slice(self) -> None:
        arr = np.arange(10, dtype=np.float64)
        strided = arr[::2]
        assert not strided.flags["C_CONTIGUOUS"]
        msg = (
            "Conversion of non-contiguous strided tensors is not yet supported"
        )
        with pytest.raises(ValueError, match=msg):
            ExternalAllocation.from_dlpack(strided)

    def test_accepts_non_monotonic_strides(self) -> None:
        arr = np.arange(2 * 3 * 4, dtype=np.float64).reshape(
            2, 3, 4
        )  # strides in bytes : (96, 32, 8)
        permuted = arr.transpose(0, 2, 1)  # strides in bytes : (96, 8, 32)
        assert not permuted.flags["C_CONTIGUOUS"]
        assert not permuted.flags["F_CONTIGUOUS"]
        alloc = ExternalAllocation.from_dlpack(permuted)
        assert alloc.size == permuted.nbytes

    def test_accepts_fortran_contiguous(self) -> None:
        arr = np.arange(12, dtype=np.float64).reshape(3, 4).copy(order="F")
        assert arr.flags["F_CONTIGUOUS"]
        alloc = ExternalAllocation.from_dlpack(arr)
        assert alloc.size == arr.nbytes

    def test_accepts_unversioned_capsule(self) -> None:
        buf = np.arange(10, dtype=np.float64)
        alloc = ExternalAllocation.from_dlpack(
            _UnversionedDLPack(buf), read_only=True
        )
        assert alloc.size == buf.nbytes
        assert alloc.read_only is True


class TestFromSysmem:
    def test_basic(self) -> None:
        buf = np.arange(10, dtype=np.float64)
        alloc = ExternalAllocation.from_sysmem(
            buf.ctypes.data, buf.nbytes, read_only=True
        )
        assert alloc.size == buf.nbytes
        assert alloc.read_only is True

    def test_null_pointer(self) -> None:
        with pytest.raises(ValueError, match="null"):
            ExternalAllocation.from_sysmem(0, 100, read_only=True)

    def test_source_ref(self) -> None:
        buf = np.zeros(10, dtype=np.float64)
        refcount_before = sys.getrefcount(buf)
        alloc = ExternalAllocation.from_sysmem(
            buf.ctypes.data, buf.nbytes, read_only=True, source=buf
        )
        refcount_after = sys.getrefcount(buf)
        assert refcount_after > refcount_before
        del alloc


@pytest.mark.skipif(
    not install_info.use_cuda,
    reason="not severe: ZCMEM requires a CUDA-enabled build",
)
class TestFromZcmem:
    def test_basic(self) -> None:
        buf = np.arange(10, dtype=np.float64)
        alloc = ExternalAllocation.from_zcmem(
            buf.ctypes.data, buf.nbytes, read_only=True, source=buf
        )
        assert alloc.size == buf.nbytes
        assert alloc.read_only is True


@pytest.mark.skipif(
    cupy is None
    or get_legate_runtime().get_machine().only(TaskTarget.GPU).empty,
    reason="not severe: test requires cupy and GPUs",
)
class TestFromFbmem:
    def test_basic(self) -> None:
        buf = cupy.zeros(10, dtype=cupy.float64)
        ptr = buf.data.ptr
        alloc = ExternalAllocation.from_fbmem(
            device_id=0, ptr=ptr, size=buf.nbytes, read_only=True
        )
        assert alloc.size == buf.nbytes
        assert alloc.read_only is True

    def test_null_pointer(self) -> None:
        with pytest.raises(ValueError, match="null"):
            ExternalAllocation.from_fbmem(
                device_id=0, ptr=0, size=100, read_only=True
            )
