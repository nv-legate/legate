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

from .utils import tasks, utils
from .utils.data import (
    ARRAY_TYPES,
    BROADCAST_SHAPES,
    EMPTY_SHAPES,
    SCALAR_VALS,
    SHAPES,
)


class TestStoreOps:
    @pytest.mark.parametrize("shape", SHAPES + EMPTY_SHAPES, ids=str)
    @pytest.mark.parametrize("dtype", ARRAY_TYPES, ids=str)
    def test_issue_copy_to_buffer(
        self, shape: tuple[int, ...], dtype: ty.Type
    ) -> None:
        runtime = get_legate_runtime()
        arr_np, store = utils.create_np_array_and_store(dtype, shape)
        out_np, out = utils.zero_array_and_store(dtype, shape=shape)
        runtime.issue_copy(out, store)
        np.testing.assert_allclose(out_np, arr_np)

    @pytest.mark.xfail(reason="issue-818: SIGSEGV", run=False)
    @pytest.mark.parametrize("shape", [(2, 1, 3), (2, 1024, 3)], ids=str)
    def test_issue_copy_from_point_type(self, shape: tuple[int, ...]) -> None:
        runtime = get_legate_runtime()
        dtype = ty.point_type(LEGATE_MAX_DIM)
        arr_np, store = utils.create_np_array_and_store(dtype, shape)
        out_np, out = utils.zero_array_and_store(dtype, shape=shape)
        # issue-818: SEGV during issue_copy
        # if the store has a rather small size like (2, 1, 3) it can pass
        # from time to time, but the test suite will get SIGSEGV at random
        # places
        runtime.issue_copy(out, store)
        np.testing.assert_allclose(out_np, arr_np)

    def test_issue_copy_with_redop(self) -> None:
        shape = (1, 3, 1)
        dtype = ty.float64
        runtime = get_legate_runtime()
        arr_np, store = utils.random_array_and_store(shape)
        out_np, out = utils.zero_array_and_store(dtype, shape=shape)
        exp_np = arr_np + out_np
        runtime.issue_copy(out, store, ty.ReductionOpKind.ADD)
        np.testing.assert_allclose(out_np, exp_np)

    @pytest.mark.xfail(run=False, reason="issue-733: hangs or segfaults")
    @pytest.mark.parametrize("np_size", [9.0, 9], ids=str)
    def test_copy_with_redop_mixed_buffer_shape(self, np_size: float) -> None:
        shape = (3, 3)

        runtime = get_legate_runtime()
        # issue-733: test process hangs or segfaults when arr_np has a
        # different dtype to store. When np_size == 9.0 this passes but will
        # still dump traces during free()
        arr_np = np.arange(np_size).reshape(shape)
        out_np = np.arange(3.0)
        store = runtime.create_store_from_buffer(
            ty.float64, shape, arr_np, False
        )
        out = runtime.create_store_from_buffer(
            ty.float64, shape, out_np, False
        )
        # recreate the np arrays to get expected muliply result since the
        # stores have different shapes to the original arrays
        arr_np = np.asarray(store.get_physical_store().get_inline_allocation())
        out_np = np.asarray(out.get_physical_store().get_inline_allocation())
        exp_np = arr_np * out_np
        runtime.issue_copy(out, store, ty.ReductionOpKind.MUL)
        np.testing.assert_allclose(out_np, exp_np)

    @pytest.mark.parametrize("src_shape, tgt_shape", BROADCAST_SHAPES, ids=str)
    def test_issue_gather(
        self, src_shape: tuple[int, ...], tgt_shape: tuple[int, ...]
    ) -> None:
        runtime = get_legate_runtime()
        arr_np, store = utils.random_array_and_store(src_shape)
        out_np, out = utils.zero_array_and_store(ty.float64, tgt_shape)
        ndim = len(src_shape)
        ind = runtime.create_store(ty.point_type(ndim), tgt_shape)
        runtime.issue_fill(ind, Scalar((0,) * ndim, ty.point_type(ndim)))
        runtime.issue_gather(out, store, ind)
        assert (
            np.isin(out_np, arr_np).ravel()[: min(len(out_np), len(arr_np))]
        ).all()

    @pytest.mark.parametrize("src_shape, tgt_shape", BROADCAST_SHAPES, ids=str)
    def test_issue_scatter(
        self, src_shape: tuple[int, ...], tgt_shape: tuple[int, ...]
    ) -> None:
        runtime = get_legate_runtime()
        arr_np, store = utils.random_array_and_store(src_shape)
        out_np, out = utils.zero_array_and_store(ty.float64, tgt_shape)

        ndim = len(tgt_shape)
        ind = runtime.create_store(ty.point_type(ndim), src_shape)
        runtime.issue_fill(ind, Scalar((0,) * ndim, ty.point_type(ndim)))
        runtime.issue_scatter(out, ind, store)
        assert np.isin(out_np, arr_np).ravel()[0]

    @pytest.mark.parametrize("src_shape, tgt_shape", BROADCAST_SHAPES, ids=str)
    def test_issue_scatter_gather(
        self, src_shape: tuple[int, ...], tgt_shape: tuple[int, ...]
    ) -> None:
        runtime = get_legate_runtime()
        arr_np, store = utils.random_array_and_store(src_shape)
        out_np, out = utils.zero_array_and_store(ty.float64, tgt_shape)

        src_ndim = len(src_shape)
        tgt_ndim = len(tgt_shape)

        src_ind = utils.create_initialized_store(
            ty.point_type(src_ndim),
            tgt_shape,
            Scalar((0,) * src_ndim, ty.point_type(src_ndim)),
        )
        tgt_ind = utils.create_initialized_store(
            ty.point_type(tgt_ndim),
            tgt_shape,
            Scalar((0,) * tgt_ndim, ty.point_type(tgt_ndim)),
        )
        runtime.issue_scatter_gather(out, tgt_ind, store, src_ind)
        assert np.isin(out_np, arr_np).ravel()[0]

    @pytest.mark.parametrize(
        "dtype, val", zip(ARRAY_TYPES, SCALAR_VALS), ids=str
    )
    @pytest.mark.parametrize("create", [True, False])
    def test_issue_fill_scalar(
        self, dtype: ty.Type, val: Any, create: bool
    ) -> None:
        shape = range(1, LEGATE_MAX_DIM + 1)
        runtime = get_legate_runtime()
        store = runtime.create_store(dtype, shape)
        scalar = Scalar(val, dtype) if create else val
        runtime.issue_fill(store, scalar)
        arr = np.asarray(store.get_physical_store().get_inline_allocation())
        assert (arr == scalar).all()
        assert (arr == val).all()

    @pytest.mark.parametrize(
        "dtype, val", zip(ARRAY_TYPES, SCALAR_VALS), ids=str
    )
    def test_issue_fill_store(self, dtype: ty.Type, val: Any) -> None:
        runtime = get_legate_runtime()
        arr, store = utils.create_np_array_and_store(
            dtype, tuple(range(1, LEGATE_MAX_DIM + 1))
        )
        val_store = runtime.create_store_from_scalar(Scalar(val, dtype))
        runtime.issue_fill(store, val_store)
        assert (arr == val).all()

    @pytest.mark.skip(reason="not implemented yet")
    def test_tree_reduce(self) -> None:
        # TODO(wonchanl): can't bind buffer to unbound PhysicalStore from
        # cython yet
        pass


class TestArrayOps:
    @pytest.mark.parametrize(
        "dtype, val", zip(ARRAY_TYPES, SCALAR_VALS), ids=str
    )
    @pytest.mark.parametrize("create", [True, False])
    def test_issue_fill_scalar(
        self, dtype: ty.Type, val: Any, create: bool
    ) -> None:
        shape = range(1, LEGATE_MAX_DIM + 1)
        runtime = get_legate_runtime()
        lg_arr = runtime.create_array(dtype, shape)
        scalar = Scalar(val, dtype) if create else val
        runtime.issue_fill(lg_arr, scalar)
        np_arr = np.asarray(lg_arr.get_physical_array())
        assert (np_arr == scalar).all()
        assert (np_arr == val).all()

    @pytest.mark.parametrize(
        "dtype, val", zip(ARRAY_TYPES, SCALAR_VALS), ids=str
    )
    def test_issue_fill_store(self, dtype: ty.Type, val: Any) -> None:
        runtime = get_legate_runtime()
        lg_arr = runtime.create_array(
            dtype, tuple(range(1, LEGATE_MAX_DIM + 1))
        )
        val_store = runtime.create_store_from_scalar(Scalar(val, dtype))
        runtime.issue_fill(lg_arr, val_store)
        np_arr = np.asarray(lg_arr.get_physical_array())
        assert (np_arr == val).all()

    def test_issue_fill_np_array(self) -> None:
        runtime = get_legate_runtime()
        lg_arr = runtime.create_array(
            ty.array_type(ty.float64, LEGATE_MAX_DIM), (3, 1, 3)
        )
        val = np.random.rand(LEGATE_MAX_DIM)
        runtime.issue_fill(lg_arr, val)
        np_arr = np.asarray(lg_arr.get_physical_array())
        np.testing.assert_allclose(np.frombuffer(np_arr[0, 0, 0]), val)
        assert np.unique(np_arr).size == 1


class TestStoreOpsErrors:
    def test_issue_copy_shape_mismatch(self) -> None:
        runtime = get_legate_runtime()
        source = runtime.create_store(ty.int8, (1, 2, 3))
        target = runtime.create_store(ty.int8, (1, 2))
        msg = "Alignment requires the stores to have the same shape"
        with pytest.raises(ValueError, match=msg):
            runtime.issue_copy(target, source)

    def test_issue_copy_type_mismatch(self) -> None:
        runtime = get_legate_runtime()
        source = runtime.create_store(ty.int8, (1, 2, 3))
        target = runtime.create_store(ty.float32, (1, 2, 3))
        msg = "Source and target must have the same type"
        with pytest.raises(ValueError, match=msg):
            runtime.issue_copy(target, source)

    def test_copy_invalid_redop(self) -> None:
        runtime = get_legate_runtime()
        source = runtime.create_store(ty.float32, (1, 2, 3))
        target = runtime.create_store(ty.float32, (1, 2, 3))
        msg = "Reduction op .* does not exist"
        with pytest.raises(ValueError, match=msg):
            runtime.issue_copy(target, source, 255)

    def test_issue_gather_invalid_ind_store(self) -> None:
        dtype = ty.float64
        shape = (3, 1, 3)
        runtime = get_legate_runtime()
        store = runtime.create_store(dtype, shape)
        out = runtime.create_store(dtype, shape)
        msg = "Indirection store should contain.*points"
        with pytest.raises(ValueError, match=msg):
            runtime.issue_gather(out, store, store)

    def test_issue_fill_non_scalar(self) -> None:
        runtime = get_legate_runtime()
        store = runtime.create_store(ty.float64, (1, 1, 1))
        val = runtime.create_store(ty.float64, (1, 1))
        msg = "Fill value should be a Future-back store"
        with pytest.raises(ValueError, match=msg):
            runtime.issue_fill(store, val)

    def test_issue_fill_mismatching_dtype(self) -> None:
        shape = range(1, LEGATE_MAX_DIM + 1)
        runtime = get_legate_runtime()
        store = runtime.create_store(ty.float64, shape)
        val = Scalar(np.random.random(), ty.float32)
        msg = "Fill value and target must have the same type"
        with pytest.raises(ValueError, match=msg):
            runtime.issue_fill(store, val)

    def test_tree_reduce_ndarray(self) -> None:
        runtime = get_legate_runtime()
        store = runtime.create_store(ty.float64, (1, 1))
        msg = "Multi-dimensional stores are not supported"
        with pytest.raises(RuntimeError, match=msg):
            runtime.tree_reduce(
                runtime.core_library, tasks.zeros_task.task_id, store
            )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
