# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import pytest

from legate.core import from_dlpack, types as ty

if TYPE_CHECKING:
    from legate.core import Type


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
