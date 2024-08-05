# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

import numpy as np
import pytest

import legate.core.types as ty
from legate.core import LEGATE_MAX_DIM, LogicalArray, get_legate_runtime

from .utils.data import ARRAY_TYPES, EMPTY_SHAPES, SHAPES
from .utils.utils import random_array_and_store


class TestArrayCreation:
    @pytest.mark.parametrize("shape", SHAPES + EMPTY_SHAPES, ids=str)
    @pytest.mark.parametrize("dtype", ARRAY_TYPES, ids=str)
    def test_create_array_like(
        self, shape: tuple[int, ...], dtype: ty.Type
    ) -> None:
        runtime = get_legate_runtime()
        np_arr0, store = random_array_and_store(shape)
        lg_arr1 = LogicalArray.from_store(store)
        np_arr1 = np.asarray(lg_arr1.get_physical_array())
        lg_arr2 = runtime.create_array_like(lg_arr1, dtype)
        np_arr2 = np.asarray(lg_arr2.get_physical_array())
        # no sure what else to assert on
        assert np_arr2.shape == np_arr1.shape
        assert lg_arr2.type == dtype
        assert not lg_arr2.data.equal_storage(lg_arr1.data)
        np.testing.assert_allclose(np_arr1, np_arr0)


class TestPromote:
    @pytest.mark.skip(
        reason="issue 498, "
        "promoted.data.get_physical_store().get_inline_allocation() crashed"
    )
    @pytest.mark.parametrize("dtype", ARRAY_TYPES)
    @pytest.mark.parametrize("nullable", [True, False])
    def test_dtype(self, dtype: ty.Type, nullable: bool) -> None:
        runtime = get_legate_runtime()
        arr_np = np.ndarray(dtype=dtype.to_numpy_dtype(), shape=(1, 2, 3, 4))
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

    @pytest.mark.skip(
        reason="issue 498, "
        "promoted.data.get_physical_store().get_inline_allocation() crashed"
    )
    @pytest.mark.parametrize("shape", SHAPES + EMPTY_SHAPES)
    @pytest.mark.parametrize("nullable", [True, False])
    def test_shape(self, shape: tuple[int, ...], nullable: bool) -> None:
        runtime = get_legate_runtime()
        arr_np = np.ndarray(dtype=np.int64, shape=shape)
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
        "arr_shape, axes",
        [
            ((2, 2), (1, 0)),
            ((3, 2, 12), (2, 1, 0)),
            ((1024,), (0,)),
            ((1024, 0, 1), (0, 2, 1)),
            ((1, 2, 4, 8), (3, 0, 2, 1)),
            (
                range(1, LEGATE_MAX_DIM + 1),
                range(LEGATE_MAX_DIM),
            ),
            (
                range(LEGATE_MAX_DIM),
                range(LEGATE_MAX_DIM),
            ),
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


class TestArrayCreationErrors:
    @pytest.mark.parametrize(
        "dtype", [ty.string_type, ty.struct_type([ty.int8])]
    )
    def test_create_array_like_invalid_dtype(self, dtype):
        runtime = get_legate_runtime()
        arr = runtime.create_array(dtype, (1,))
        msg = "doesn't support variable size types or struct types"
        with pytest.raises(RuntimeError, match=msg):
            runtime.create_array_like(arr, arr.type)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
