# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import pytest

from legate.core import (
    LEGATE_MAX_DIM,
    LogicalArray,
    StoreTarget,
    TaskTarget,
    Type,
    get_legate_runtime,
    types as ty,
)

from .utils.data import ARRAY_TYPES, EMPTY_SHAPES, SHAPES
from .utils.utils import random_array_and_store, zero_array_and_store


class TestArrayCreation:
    @pytest.mark.parametrize("shape", SHAPES + EMPTY_SHAPES, ids=str)
    @pytest.mark.parametrize("dtype", (*ARRAY_TYPES, None), ids=str)
    def test_create_array_like(
        self, shape: tuple[int, ...], dtype: None | ty.Type
    ) -> None:
        runtime = get_legate_runtime()
        np_arr0, store = random_array_and_store(shape)
        lg_arr1 = LogicalArray.from_store(store)
        np_arr1 = np.asarray(lg_arr1)
        lg_arr2 = runtime.create_array_like(lg_arr1, dtype)
        np_arr2 = np.asarray(lg_arr2)
        # not sure what else to assert on
        assert np_arr2.shape == np_arr1.shape
        if dtype is None:
            assert lg_arr2.type.to_numpy_dtype() == np_arr1.dtype
        else:
            assert lg_arr2.type == dtype
        assert not lg_arr2.data.equal_storage(lg_arr1.data)
        np.testing.assert_allclose(np_arr1, np_arr0)
        assert lg_arr2.volume == lg_arr1.volume

    @pytest.mark.parametrize("shape", SHAPES + EMPTY_SHAPES, ids=str)
    def test_nullable_array_constructor(self, shape: tuple[int, ...]) -> None:
        _, data_store = random_array_and_store(shape)
        _, mask_store = zero_array_and_store(ty.bool_, shape)
        arr = LogicalArray.from_store_and_mask(data_store, mask_store)

        assert arr.nullable
        assert arr.null_mask.shape == shape
        assert arr.null_mask.type == ty.bool_
        assert arr.null_mask.equal_storage(mask_store)
        assert arr.data.equal_storage(data_store)

    @pytest.mark.parametrize("shape", SHAPES + EMPTY_SHAPES, ids=str)
    def test_runtime_create_nullable_array(
        self, shape: tuple[int, ...]
    ) -> None:
        runtime = get_legate_runtime()
        _, data_store = random_array_and_store(shape)
        _, mask_store = zero_array_and_store(ty.bool_, shape)
        arr = runtime.create_nullable_array(data_store, mask_store)

        assert arr.nullable
        assert arr.null_mask.shape == shape
        assert arr.null_mask.type == ty.bool_
        assert arr.null_mask.equal_storage(mask_store)
        assert arr.data.equal_storage(data_store)

    @pytest.mark.skipif(
        get_legate_runtime().machine.preferred_target != TaskTarget.GPU,
        reason="FBMEM target only works with GPU",
    )
    def test_fbmem_target(self) -> None:
        # Need to explicitly check cupy existence here
        try:
            import cupy  # type: ignore[import-not-found]  # noqa: PLC0415
        except ModuleNotFoundError:
            pytest.skip(reason="Test requires cupy to be installed")
        runtime = get_legate_runtime()
        lg_arr = runtime.create_array(ty.int32, shape=(3, 3, 3))
        lg_arr.fill(1)
        np_arr = np.ones((3, 3, 3), np.int32)
        assert np.allclose(cupy.asarray(lg_arr), np_arr)
        # not sure what can be validated, just check the call returns alright
        # for code coverage
        lg_arr.offload_to(StoreTarget.SYSMEM)


class TestPromote:
    @pytest.mark.parametrize("dtype", ARRAY_TYPES, ids=str)
    def test_dtype(self, dtype: ty.Type) -> None:
        runtime = get_legate_runtime()
        arr_np = np.empty(dtype=dtype.to_numpy_dtype(), shape=(1, 2, 3, 4))
        store = runtime.create_store_from_buffer(
            dtype, arr_np.shape, arr_np, False
        )
        arr_logical = LogicalArray.from_store(store)

        expanded_arr = np.expand_dims(arr_np, 3)
        promoted = arr_logical.promote(3, 1)
        promoted_arr = np.asarray(
            promoted.data.get_physical_store().get_inline_allocation()
        )

        assert promoted_arr.ndim == expanded_arr.ndim
        assert promoted_arr.shape == expanded_arr.shape
        assert promoted_arr.dtype == expanded_arr.dtype

    @pytest.mark.parametrize("shape", SHAPES + EMPTY_SHAPES, ids=str)
    def test_shape(self, shape: tuple[int, ...]) -> None:
        if len(shape) >= LEGATE_MAX_DIM:
            pytest.skip(reason="promote exceeds max dim")
        runtime = get_legate_runtime()
        arr_np = np.empty(dtype=np.int64, shape=shape)
        store = runtime.create_store_from_buffer(
            ty.int64, arr_np.shape, arr_np, False
        )
        arr_logical = LogicalArray.from_store(store)

        expand_dim = arr_np.ndim - 1
        expanded_arr = np.expand_dims(arr_np, expand_dim)
        promoted = arr_logical.promote(expand_dim, 1)

        promoted_arr = np.asarray(
            promoted.data.get_physical_store().get_inline_allocation()
        )
        assert promoted_arr.ndim == expanded_arr.ndim
        assert promoted_arr.shape == expanded_arr.shape
        assert promoted_arr.dtype == expanded_arr.dtype


class TestTranspose:
    @pytest.mark.parametrize(
        ("arr_shape", "axes"),
        [
            ((2, 2), (1, 0)),
            ((3, 2, 12), (2, 1, 0)),
            ((1024,), (0,)),
            ((1024, 0, 1), (0, 2, 1)),
            ((1, 2, 4, 8), (3, 0, 2, 1)),
            (range(1, LEGATE_MAX_DIM + 1), range(LEGATE_MAX_DIM)),
            (range(LEGATE_MAX_DIM), range(LEGATE_MAX_DIM)),
        ],
        ids=str,
    )
    def test_basic(
        self, arr_shape: tuple[int, ...], axes: tuple[int, ...]
    ) -> None:
        runtime = get_legate_runtime()
        arr = np.random.random(arr_shape)
        store = runtime.create_store_from_buffer(
            ty.float64, arr.shape, arr, False
        )
        logical_arr = LogicalArray.from_store(store)

        arr_t = arr.transpose(axes)
        logical_t = logical_arr.transpose(axes)
        assert arr_t.shape == logical_t.shape
        assert arr_t.ndim == logical_t.ndim

        logical_t_data = np.asarray(
            logical_t.data.get_physical_store().get_inline_allocation()
        )
        arr_t_data = np.asarray(arr_t)
        assert np.array_equal(arr_t_data, logical_t_data, equal_nan=True)

    def test_nullable(self) -> None:
        dtype = ty.int32
        runtime = get_legate_runtime()
        arr = runtime.create_array(dtype=dtype, shape=(1, 2, 3), nullable=True)
        assert not arr.unbound
        assert arr.ndim == 3
        assert arr.shape == (1, 2, 3)
        assert arr.type == dtype
        assert arr.nullable
        assert not arr.nested
        assert arr.num_children == 0
        assert arr.null_mask.shape == arr.shape


class TestArrayCreationErrors:
    @pytest.mark.parametrize(
        "dtype", [ty.string_type, ty.struct_type([ty.int8])]
    )
    def test_create_array_like_invalid_dtype(self, dtype: Type) -> None:
        runtime = get_legate_runtime()
        arr = runtime.create_array(dtype, (1,))
        msg = "doesn't support variable size types or struct types"
        with pytest.raises(RuntimeError, match=msg):
            runtime.create_array_like(arr, arr.type)

    def test_nullable_array_interface(self) -> None:
        runtime = get_legate_runtime()
        arr = runtime.create_array(ty.uint16, shape=(1,), nullable=True)
        msg = "nullable arrays don't support the array interface directly"
        with pytest.raises(ValueError, match=msg):
            np.asarray(arr.get_physical_array())

    def test_runtime_create_nullable_array_invalid_type(self) -> None:
        runtime = get_legate_runtime()
        _, data_store = random_array_and_store(shape=(1,))
        _, mask_store = zero_array_and_store(ty.float64, shape=(1,))
        with pytest.raises(
            ValueError, match="Null mask must be a boolean type"
        ):
            runtime.create_nullable_array(data_store, mask_store)

    def test_runtime_create_nullable_array_invalid_shape(self) -> None:
        runtime = get_legate_runtime()
        _, data_store = random_array_and_store(shape=(1,))
        _, mask_store = zero_array_and_store(ty.bool_, shape=(2,))
        with pytest.raises(
            ValueError, match="Store and null mask must have the same shape"
        ):
            runtime.create_nullable_array(data_store, mask_store)

    def test_runtime_create_nullable_array_not_top_level(self) -> None:
        runtime = get_legate_runtime()
        _, data_store = random_array_and_store(shape=(2, 2))
        _, mask_store = zero_array_and_store(ty.bool_, shape=(2, 2))
        data_store = data_store.transpose((1, 0))
        with pytest.raises(
            ValueError, match="Store and null mask must be top-level stores"
        ):
            runtime.create_nullable_array(data_store, mask_store)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
