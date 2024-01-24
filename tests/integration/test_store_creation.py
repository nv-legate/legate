# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
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

from typing import Any

import numpy as np
import pytest

from legate.core import LEGATE_MAX_DIM, Scalar, get_legate_runtime, types as ty

ARRAY_TYPES = (
    ty.bool_,
    ty.complex128,
    ty.complex64,
    ty.float16,
    ty.float32,
    ty.float64,
    ty.int16,
    ty.int32,
    ty.int64,
    ty.int8,
    ty.uint16,
    ty.uint32,
    ty.uint64,
    ty.uint8,
)

VALS = (
    True,
    complex(1, 5),
    complex(5, 1),
    12.5,
    3.1415,
    0.7777777,
    10,
    1024,
    4096,
    -1,
    65535,
    4294967295,
    101010,
)


class TestStoreCreation:
    @pytest.mark.parametrize(
        "val, dtype",
        zip(VALS, ARRAY_TYPES),
        ids=str,
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
        assert np.allclose(arr_np, arr_store)

    @pytest.mark.xfail(run=False, reason="accessing inline allocation crashes")
    def test_create_from_null_scalar(self) -> None:
        runtime = get_legate_runtime()
        scalar = Scalar.null()
        store = runtime.create_store_from_scalar(scalar)
        assert store.has_scalar_storage
        assert store.ndim == 1
        assert not store.unbound

        # crashes the application
        # [error 87] LEGION ERROR: Future size mismatch!
        # Expected non-empty future for making an accessor but future has a
        # payload of 0 bytes. (UID 0)
        store.get_physical_store().get_inline_allocation()

    @pytest.mark.parametrize("dtype", ARRAY_TYPES, ids=str)
    def test_store_dtype(self, dtype: ty.Type) -> None:
        runtime = get_legate_runtime()
        store = runtime.create_store(dtype=dtype, shape=(1, 2, 3))
        arr_np = np.zeros(dtype=dtype.to_numpy_dtype(), shape=(1, 2, 3))
        arr_store = np.asarray(
            store.get_physical_store().get_inline_allocation()
        )
        assert arr_np.dtype == arr_store.dtype
        assert np.allclose(arr_np, arr_store)

    @pytest.mark.parametrize(
        "shape",
        [
            (1, 2, 3),
            (6,),
            (1, 1, 1, 1),
            (4096, 1, 5),
        ],
        ids=str,
    )
    def test_store_shape(self, shape: tuple[int]) -> None:
        runtime = get_legate_runtime()
        store = runtime.create_store(dtype=ty.int32, shape=shape)
        arr_np = np.zeros(dtype=np.int32, shape=shape)
        arr_store = np.asarray(
            store.get_physical_store().get_inline_allocation()
        )
        assert arr_np.shape == arr_store.shape
        assert np.allclose(arr_np, arr_store)

    def test_create_from_numpy_array(self) -> None:
        runtime = get_legate_runtime()
        arr_np = np.random.random((4, 2, 3))
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
        msg = "shape must be a Shape object or an iterable"
        with pytest.raises(ValueError, match=msg):
            runtime.create_store(ty.int32, shape=1)

    def test_invalid_shape_type(self) -> None:
        runtime = get_legate_runtime()
        msg = "an integer is required"
        with pytest.raises(TypeError, match=msg):
            runtime.create_store(ty.int32, shape=("a", "b"))

    def test_exceed_max_dim(self) -> None:
        runtime = get_legate_runtime()
        msg = (
            "The maximum number of dimensions is 4, "
            "but a 5-D store is requested"
        )
        with pytest.raises(IndexError, match=msg):
            runtime.create_store(ty.int32, shape=(1,) * (LEGATE_MAX_DIM + 1))

    def test_string_scalar(self) -> None:
        runtime = get_legate_runtime()
        msg = "Store cannot be created with variable size type string"
        scalar = Scalar("abcd", ty.string_type)
        with pytest.raises(ValueError, match=msg):
            runtime.create_store_from_scalar(scalar)

    def test_null_type_store_array_interface(self) -> None:
        runtime = get_legate_runtime()
        msg = "Invalid type code: 15"
        store = runtime.create_store(dtype=ty.null_type, shape=(1,))
        with pytest.raises(ValueError, match=msg):
            np.asarray(store.get_physical_store().get_inline_allocation())

    def test_get_unbound_physical_store(self) -> None:
        runtime = get_legate_runtime()
        msg = "Unbound store cannot be inlined mapped"
        store = runtime.create_store(ty.int64)
        with pytest.raises(ValueError, match=msg):
            store.get_physical_store()
