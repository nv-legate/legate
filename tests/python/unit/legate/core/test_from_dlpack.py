# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import re
import ctypes
from typing import TYPE_CHECKING, Any

import numpy as np

import pytest

from legate.core import from_dlpack, types as ty

if TYPE_CHECKING:
    from typing_extensions import CapsuleType

    from legate.core import Type

try:
    import cupy  # type: ignore[import-not-found]
except ModuleNotFoundError:
    cupy = None


class TestFromDLPack:
    @pytest.mark.parametrize("shape", ((1,), (2, 2), (3, 3, 3)))
    @pytest.mark.parametrize(
        "legate_type",
        (ty.int32, ty.float32, ty.bool_, ty.uint64, ty.complex128),
    )
    def test_numpy(self, shape: tuple[int, ...], legate_type: Type) -> None:
        x = np.ones(shape=shape, dtype=legate_type.to_numpy_dtype())
        store = from_dlpack(x)

        assert store.shape == x.shape
        assert store.ndim == len(shape)
        assert store.type == legate_type
        assert store.type.to_numpy_dtype() == x.dtype
        assert store.size == x.size

        phys = store.get_physical_store()
        x_phys = np.asarray(phys)
        assert (x == x_phys).all()

    @pytest.mark.parametrize("shape", ((1,), (2, 2), (3, 3, 3)))
    @pytest.mark.parametrize(
        "legate_type", (ty.int32, ty.float32, ty.uint64, ty.complex128)
    )
    def test_numpy_force_copy(
        self, shape: tuple[int, ...], legate_type: Type
    ) -> None:
        x = np.ones(shape=shape, dtype=legate_type.to_numpy_dtype())
        store = from_dlpack(x, copy=True)

        assert store.shape == x.shape
        assert store.ndim == len(shape)
        assert store.type == legate_type
        assert store.type.to_numpy_dtype() == x.dtype
        assert store.size == x.size

        phys = store.get_physical_store()
        x_phys = np.asarray(phys)
        new_val = x_phys.dtype.type(34)
        x_phys[:] = new_val

        assert (x_phys == new_val).all()
        assert (x == 1).all()
        assert (x_phys != x).all()

    @pytest.mark.parametrize("shape", ((1,), (2, 2), (3, 3, 3)))
    @pytest.mark.parametrize(
        "legate_type", (ty.int32, ty.float32, ty.uint64, ty.complex128)
    )
    def test_numpy_never_copy(
        self, shape: tuple[int, ...], legate_type: Type
    ) -> None:
        x = np.ones(shape=shape, dtype=legate_type.to_numpy_dtype())
        store = from_dlpack(x, copy=False)

        assert store.shape == x.shape
        assert store.ndim == len(shape)
        assert store.type == legate_type
        assert store.type.to_numpy_dtype() == x.dtype
        assert store.size == x.size

        phys = store.get_physical_store()
        x_phys = np.asarray(phys)
        new_val = x_phys.dtype.type(34)
        x_phys[:] = new_val

        assert (x_phys == new_val).all()
        assert (x == new_val).all()
        assert (x_phys == x).all()

    @pytest.mark.parametrize("shape", ((1,), (2, 2), (3, 3, 3)))
    @pytest.mark.parametrize(
        "legate_type", (ty.int32, ty.float32, ty.uint64, ty.complex128)
    )
    def test_numpy_fortran_order(
        self, shape: tuple[int, ...], legate_type: Type
    ) -> None:
        x = np.ones(shape=shape, dtype=legate_type.to_numpy_dtype(), order="F")
        assert x.flags["F_CONTIGUOUS"]
        store = from_dlpack(x)

        assert store.shape == x.shape
        assert store.ndim == len(shape)
        assert store.type == legate_type
        assert store.type.to_numpy_dtype() == x.dtype
        assert store.size == x.size

        phys = store.get_physical_store()
        x_phys = np.asarray(phys)
        new_val = x_phys.dtype.type(34)
        x_phys[:] = new_val

        assert (x_phys == new_val).all()
        assert (x == new_val).all()
        assert (x_phys == x).all()

    @pytest.mark.parametrize("shape", ((1,), (2, 2), (3, 3, 3)))
    @pytest.mark.parametrize(
        "legate_type", (ty.int32, ty.float32, ty.uint64, ty.complex128)
    )
    def test_from_store_from_numpy(
        self, shape: tuple[int, ...], legate_type: Type
    ) -> None:
        x = np.ones(shape=shape, dtype=legate_type.to_numpy_dtype())
        store1 = from_dlpack(x)
        store2 = from_dlpack(store1.get_physical_store())
        store1_arr = np.asarray(store1.get_physical_store())
        store2_arr = np.asarray(store2.get_physical_store())
        np.testing.assert_allclose(store1_arr, x)
        np.testing.assert_allclose(store2_arr, x)

    @pytest.mark.xfail(run=False, reason="aborts python proc")
    @pytest.mark.parametrize("shape", ((1,), (2, 2), (3, 3, 3)))
    @pytest.mark.parametrize(
        "legate_type", (ty.int32, ty.float32, ty.uint64, ty.complex128)
    )
    def test_from_store_from_cupy(
        self, shape: tuple[int, ...], legate_type: Type
    ) -> None:
        if not cupy:
            pytest.skip(reason="test requires cupy")

        x = cupy.ones(shape=shape, dtype=legate_type.to_numpy_dtype())
        store1 = from_dlpack(x)
        store2 = from_dlpack(store1.get_physical_store())
        store1_arr = cupy.asarray(store1.get_physical_store())
        # TODO(yimoj) [issue-2730]
        # LEGATE ERROR: #0 Legate called abort at mapping.cc:70 in to_target()
        # LEGATE ERROR: #0 Unhandled Processor::Kind 14
        store2_arr = cupy.asarray(store2.get_physical_store())
        cupy.testing.assert_allclose(store1_arr, x)
        cupy.testing.assert_allclose(store2_arr, x)


class TestFromDLPackErrors:
    def test_unhandled_device_type(self) -> None:
        x = np.empty(shape=(1,))
        device_type = (-1, 0)
        msg = f"Unhandled DLPack device type: {device_type[0]}"
        with pytest.raises(BufferError, match=msg):
            from_dlpack(x, device=device_type)

    def test_invalid_dlpack(self) -> None:
        name = b"foo_bar_baz"

        class NotDLPack:
            def __dlpack__(
                self,
                stream: int | Any | None = None,
                dl_device: tuple[int, int] | None = None,
                copy: bool | None = None,
            ) -> CapsuleType:
                PyCapsule_New = ctypes.pythonapi.PyCapsule_New
                PyCapsule_New.restype = ctypes.py_object
                PyCapsule_New.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_char_p,
                    ctypes.c_void_p,
                ]
                dummy_pointer = 1234
                return PyCapsule_New(dummy_pointer, name, None)

        msg = re.escape(
            "A DLPack tensor object cannot be consumed multiple times "
            "(or object was not a DLPack capsule)"
        )
        with pytest.raises(ValueError, match=msg):
            from_dlpack(NotDLPack())
