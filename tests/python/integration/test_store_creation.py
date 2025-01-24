# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
from __future__ import annotations

import re
from typing import Any

import numpy as np

import pytest

from legate.core import LEGATE_MAX_DIM, Scalar, get_legate_runtime, types as ty

from .utils.data import ARRAY_TYPES, SCALAR_VALS


class TestStoreCreation:
    @pytest.mark.parametrize(
        ("val", "dtype"), zip(SCALAR_VALS, ARRAY_TYPES), ids=str
    )
    def test_create_from_numpy_scalar(self, val: Any, dtype: ty.Type) -> None:
        runtime = get_legate_runtime()
        arr_np = np.array(val, dtype=dtype.to_numpy_dtype())
        scalar = Scalar(arr_np, dtype)
        store = runtime.create_store_from_scalar(scalar)
        assert store.has_scalar_storage
        assert store.ndim == 1
        arr_store = np.asarray(
            store.get_physical_store().get_inline_allocation()
        )
        if isinstance(val, bytes):
            assert (arr_np == arr_store).all()
        else:
            assert np.allclose(arr_np, arr_store)

    @pytest.mark.parametrize("dtype", ARRAY_TYPES, ids=str)
    def test_store_dtype(self, dtype: ty.Type) -> None:
        shape = (1, 2, 3)
        runtime = get_legate_runtime()
        store = runtime.create_store(dtype=dtype, shape=shape)
        arr = np.asarray(store.get_physical_store().get_inline_allocation())

        val: bool | bytes | int
        match dtype.code:
            case ty.TypeCode.BOOL:
                val = bool(0)
            case ty.TypeCode.BINARY:
                val = b""
            case _:
                val = 0

        arr.fill(val)

        exp = np.zeros(dtype=dtype.to_numpy_dtype(), shape=shape)
        if dtype.code == ty.TypeCode.BINARY:
            assert (arr == exp).all()
        else:
            np.testing.assert_allclose(arr, exp)

    @pytest.mark.parametrize(
        "shape", [(1, 2, 3), (6,), (1, 1, 1, 1), (4096, 1, 5)], ids=str
    )
    def test_store_shape(self, shape: tuple[int, ...]) -> None:
        runtime = get_legate_runtime()
        store = runtime.create_store(dtype=ty.int32, shape=shape)
        assert store.shape == shape
        arr_store = np.asarray(
            store.get_physical_store().get_inline_allocation()
        )
        arr_np = np.empty(shape=shape)
        assert arr_np.shape == arr_store.shape

    @pytest.mark.parametrize("shape", [(0,), (4, 2, 3), (1, 0, 1)], ids=str)
    def test_create_from_numpy_array(self, shape: tuple[int, ...]) -> None:
        runtime = get_legate_runtime()
        arr_np = np.random.random(shape)
        store = runtime.create_store_from_buffer(
            ty.float64, arr_np.shape, arr_np, False
        )
        arr_store = np.asarray(
            store.get_physical_store().get_inline_allocation()
        )
        assert arr_np.shape == arr_store.shape
        assert np.allclose(arr_np, arr_store)


class TestStoreCreationErrors:
    def test_invalid_shape_value(self) -> None:
        runtime = get_legate_runtime()
        msg = "Expected an iterable but got.*"
        with pytest.raises(ValueError, match=msg):
            runtime.create_store(ty.int32, shape=1)  # type: ignore[arg-type]

    def test_invalid_shape_type(self) -> None:
        runtime = get_legate_runtime()
        msg = "an integer is required"
        with pytest.raises(TypeError, match=msg):
            runtime.create_store(
                ty.int32,
                shape=("a", "b"),  # type:ignore [arg-type]
            )

    def test_exceed_max_dim(self) -> None:
        runtime = get_legate_runtime()
        with pytest.raises(IndexError, match="maximum number of dimensions"):
            runtime.create_store(ty.int32, shape=(1,) * (LEGATE_MAX_DIM + 1))

    def test_buffer_exceed_max_dim(self) -> None:
        runtime = get_legate_runtime()
        arr = np.empty(range(1, LEGATE_MAX_DIM + 2))
        with pytest.raises(IndexError, match="maximum number of dimensions"):
            runtime.create_store_from_buffer(ty.int32, arr.shape, arr, False)

    def test_string_scalar(self) -> None:
        runtime = get_legate_runtime()
        msg = re.escape("Store must have a fixed-size type")
        scalar = Scalar("abcd", ty.string_type)
        with pytest.raises(ValueError, match=msg):
            runtime.create_store_from_scalar(scalar)

    def test_get_unbound_physical_store(self) -> None:
        runtime = get_legate_runtime()
        msg = "Unbound store cannot be inlined mapped"
        store = runtime.create_store(ty.int64)
        with pytest.raises(ValueError, match=msg):
            store.get_physical_store()

    def test_small_buffer(self) -> None:
        runtime = get_legate_runtime()
        arr = np.array([1024])
        msg = "Passed buffer is too small for a store of shape .* and type .*"
        with pytest.raises(ValueError, match=msg):
            runtime.create_store_from_buffer(ty.uint64, (1024,), arr, False)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
