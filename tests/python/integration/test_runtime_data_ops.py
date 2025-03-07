# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
        arr_np, store = utils.create_np_array_and_store(
            dtype, shape=shape, func=np.ones
        )
        out_np, out = utils.create_np_array_and_store(
            dtype, shape=shape, func=np.zeros
        )
        if 0 not in shape:
            assert out_np.all() != arr_np.all()
        runtime.issue_copy(out, store)
        if dtype.code == ty.TypeCode.BINARY:
            assert (out_np == arr_np).all()
        else:
            np.testing.assert_allclose(out_np, arr_np)

    @pytest.mark.parametrize("shape", [(2, 1, 3), (2, 1024, 3)], ids=str)
    def test_issue_copy_from_point_type(self, shape: tuple[int, ...]) -> None:
        runtime = get_legate_runtime()
        dtype = ty.point_type(LEGATE_MAX_DIM)
        arr_np, store = utils.create_np_array_and_store(
            dtype, shape=shape, func=np.ones
        )
        out_np, out = utils.create_np_array_and_store(
            dtype, shape=shape, func=np.zeros
        )
        runtime.issue_copy(out, store)
        np.testing.assert_allclose(out_np, arr_np)

    def test_issue_copy_with_redop(self) -> None:
        shape = (1, 3, 1)
        dtype = ty.float64
        runtime = get_legate_runtime()
        arr_np, store = utils.create_np_array_and_store(
            dtype, shape=shape, func=np.ones
        )
        out_np, out = utils.create_np_array_and_store(
            dtype, shape=shape, func=np.zeros
        )
        exp_np = arr_np + out_np
        runtime.issue_copy(out, store, ty.ReductionOpKind.ADD)
        np.testing.assert_allclose(out_np, exp_np)

    @pytest.mark.parametrize("np_size", [9.0, 9], ids=str)
    def test_copy_with_redop_mixed_buffer_shape(self, np_size: float) -> None:
        shape = (3, 3)

        runtime = get_legate_runtime()
        arr_np = np.arange(np_size).reshape(shape).astype(np.float64)
        out_np = np.arange(np_size).astype(np.float64)
        store = runtime.create_store_from_buffer(
            ty.float64, shape, arr_np, False
        )
        out = runtime.create_store_from_buffer(
            ty.float64, shape, out_np, False
        )
        # recreate the np arrays to get expected multiply result since the
        # stores have different shapes to the original arrays
        #
        # Depending on the numpy version, mypy says:
        #
        # "ndarray[tuple[int, ...], dtype[float64]]", variable has type
        # "ndarray[tuple[int, int], dtype[float64]]")
        #
        # Be quiet mypy
        arr_np = np.asarray(  # type: ignore[assignment, unused-ignore]
            store.get_physical_store().get_inline_allocation()
        )
        out_np = np.asarray(  # type: ignore[assignment, unused-ignore]
            out.get_physical_store().get_inline_allocation()
        )
        exp_np = arr_np * out_np
        runtime.issue_copy(out, store, ty.ReductionOpKind.MUL)
        np.testing.assert_allclose(out_np, exp_np)

    @pytest.mark.parametrize(
        ("src_shape", "tgt_shape"), BROADCAST_SHAPES, ids=str
    )
    def test_issue_gather(
        self, src_shape: tuple[int, ...], tgt_shape: tuple[int, ...]
    ) -> None:
        runtime = get_legate_runtime()
        arr_np, store = utils.random_array_and_store(src_shape)
        out_np, out = utils.zero_array_and_store(ty.float64, tgt_shape)
        # The same point appearing multiple times in the indirection store
        # causes no non-determinism in gather copies, as they only do indirect
        # read accesses.
        ind_np, ind = utils.create_random_points(
            tgt_shape, src_shape, no_duplicates=False
        )
        runtime.issue_gather(out, store, ind)
        np.testing.assert_allclose(arr_np[ind_np].reshape(tgt_shape), out_np)

    def test_issue_gather_redop(self) -> None:
        src_shape = (7, 10)
        tgt_shape = (0, 0)
        runtime = get_legate_runtime()
        arr_np, store = utils.random_array_and_store(src_shape)
        out_np, out = utils.zero_array_and_store(ty.float64, tgt_shape)
        ind_np, ind = utils.create_random_points(
            tgt_shape, src_shape, no_duplicates=False
        )
        runtime.issue_gather(out, store, ind, ty.ReductionOpKind.ADD)
        np.testing.assert_allclose(arr_np[ind_np].reshape(tgt_shape), out_np)

    @pytest.mark.parametrize(
        ("tgt_shape", "src_shape"), BROADCAST_SHAPES, ids=str
    )
    def test_issue_scatter(
        self, tgt_shape: tuple[int, ...], src_shape: tuple[int, ...]
    ) -> None:
        runtime = get_legate_runtime()
        arr_np, store = utils.random_array_and_store(src_shape)
        out_np, out = utils.zero_array_and_store(ty.float64, tgt_shape)

        ind_np, ind = utils.create_random_points(
            src_shape, tgt_shape, no_duplicates=True
        )
        runtime.issue_scatter(out, ind, store)
        np.testing.assert_allclose(arr_np, out_np[ind_np].reshape(src_shape))

    def test_issue_scatter_redop(self) -> None:
        src_shape = (0, 0)
        tgt_shape = (7, 10)
        runtime = get_legate_runtime()
        arr_np, store = utils.random_array_and_store(src_shape)
        out_np, out = utils.zero_array_and_store(ty.float64, tgt_shape)

        ind_np, ind = utils.create_random_points(
            src_shape, tgt_shape, no_duplicates=True
        )
        runtime.issue_scatter(out, ind, store, ty.ReductionOpKind.ADD)
        np.testing.assert_allclose(arr_np, out_np[ind_np].reshape(src_shape))

    @pytest.mark.parametrize(
        ("tgt_shape", "src_shape"), BROADCAST_SHAPES, ids=str
    )
    def test_issue_scatter_gather(
        self, src_shape: tuple[int, ...], tgt_shape: tuple[int, ...]
    ) -> None:
        runtime = get_legate_runtime()
        arr_np, store = utils.random_array_and_store(src_shape)
        out_np, out = utils.zero_array_and_store(ty.float64, tgt_shape)

        src_ind_np, src_ind = utils.create_random_points(
            tgt_shape, src_shape, no_duplicates=False
        )
        tgt_ind_np, tgt_ind = utils.create_random_points(
            tgt_shape, tgt_shape, no_duplicates=True
        )
        runtime.issue_scatter_gather(out, tgt_ind, store, src_ind)
        np.testing.assert_allclose(arr_np[src_ind_np], out_np[tgt_ind_np])

    def test_issue_scatter_gather_redop(self) -> None:
        src_shape = tgt_shape = (10, 10)
        ind_shape = (0, 0)
        runtime = get_legate_runtime()
        arr_np, store = utils.random_array_and_store(src_shape)
        out_np, out = utils.zero_array_and_store(ty.float64, tgt_shape)

        src_ind_np, src_ind = utils.create_random_points(
            ind_shape, src_shape, no_duplicates=False
        )
        tgt_ind_np, tgt_ind = utils.create_random_points(
            ind_shape, tgt_shape, no_duplicates=True
        )
        runtime.issue_scatter_gather(
            out, tgt_ind, store, src_ind, ty.ReductionOpKind.ADD
        )
        np.testing.assert_allclose(arr_np[src_ind_np], out_np[tgt_ind_np])

    @pytest.mark.parametrize(
        ("dtype", "val"), zip(ARRAY_TYPES, SCALAR_VALS), ids=str
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

    def test_issue_fill_none(self) -> None:
        shape = range(1, LEGATE_MAX_DIM + 1)
        runtime = get_legate_runtime()
        lg_arr = runtime.create_array(ty.uint64, shape, nullable=True)
        runtime.issue_fill(lg_arr, None)
        arr = np.asarray(lg_arr.get_physical_array().data())
        assert not arr.any()

    @pytest.mark.parametrize(
        ("dtype", "val"), zip(ARRAY_TYPES, SCALAR_VALS), ids=str
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
        ("dtype", "val"), zip(ARRAY_TYPES, SCALAR_VALS), ids=str
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
        ("dtype", "val"), zip(ARRAY_TYPES, SCALAR_VALS), ids=str
    )
    def test_issue_fill_store(self, dtype: ty.Type, val: Any) -> None:
        if val is None:
            # LEGION ERROR: Fill operation 2378 in task Legate Core Toplevel
            # Task (UID 1) was launched without a fill value. All fill
            # operations must be given a non-empty argument or a future to use
            # as a fill value.
            pytest.skip()

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

    @pytest.mark.parametrize(
        ("dtype", "val"),
        [(ty.struct_type([ty.int32]), (1,)), (ty.string_type, "foo")],
        ids=str,
    )
    def test_issue_fill_unsupported(self, dtype: ty.Type, val: Any) -> None:
        runtime = get_legate_runtime()
        arr = runtime.create_array(dtype, shape=(1,))
        msg = "Fills on list or struct arrays are not supported yet"
        with pytest.raises(RuntimeError, match=msg):
            runtime.issue_fill(arr, val)

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

    def test_fill_non_nullable(self) -> None:
        runtime = get_legate_runtime()
        msg = "Non-nullable arrays cannot be filled with null"
        store = runtime.create_store(ty.float64, (1, 1))
        with pytest.raises(ValueError, match=msg):
            runtime.issue_fill(store, None)

    @pytest.mark.xfail(run=False, reason="hits LEGION ERROR and aborts proc")
    def test_prefetch_uninitialized(self) -> None:
        runtime = get_legate_runtime()
        store = runtime.create_store(ty.int32, (3, 1, 3))
        # docstring says this will trip "runtime error"
        # taking aborting python proc as correct behavior
        # LEGION ERROR: Region requirement 0 of operation
        # legate::detail::(anonymous namespace)::PrefetchBloatedInstances
        # in parent task Legate Core Toplevel Task is using uninitialized data
        # for field(s) 10000 of logical region (29,10,10)
        # with read-only privileges
        with pytest.raises(RuntimeError):
            runtime.prefetch_bloated_instances(store, (1, 2, 3), (3, 2, 1))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
