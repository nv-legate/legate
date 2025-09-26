# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import re
from typing import Any

import numpy as np

import pytest

from legate.core import LEGATE_MAX_DIM, get_legate_runtime, types as ty

from .utils import utils


class TestLogicalStoreOperation:
    @pytest.mark.parametrize(
        ("arr_shape", "dim", "shape"),
        [
            ((1, 0, 4), 1, (0,)),
            ((1024,), 0, (256, 4)),
            ((1, 2, 4), 2, (1, 4)),
            ((1, 2, 4), -2, (2,)),
            # TODO(yimoj) [issue 503]
            # Delinearize allows sizes [0,0] but SIGFPE when
            # InlineAllocation is accessed.
            # Fatal Python error: Floating point exception
            pytest.param(
                (1, 0, 1),
                1,
                (0, 0),
                marks=pytest.mark.xfail(reason="SIGFPE", run=False),
            ),
        ],
        ids=str,
    )
    def test_delinearize(
        self, arr_shape: tuple[int, ...], dim: int, shape: tuple[int, ...]
    ) -> None:
        arr, store = utils.create_np_array_and_store(ty.int16, arr_shape)
        new_shape = arr_shape[:dim] + shape + arr_shape[dim + 1 :]
        reshaped = arr.reshape(new_shape)
        delinearized = store.delinearize(dim, shape)
        assert delinearized.equal_storage(store)
        delin_arr = np.asarray(delinearized.get_physical_store())
        np.testing.assert_allclose(delin_arr, reshaped)

    @pytest.mark.parametrize(
        "shape", [(2, 4, 6, 8), (7, 4, 3, 9), (500, 2, 2, 2)], ids=str
    )
    def test_partition_by_tiling(self, shape: tuple[int, ...]) -> None:
        runtime = get_legate_runtime()
        # test might crash when there are still running tasks, block until done
        # [error 397] LEGION ERROR: Attempted an external attach operation on
        # region (d80,3a,3a) that conflicts with previous inline mapping in
        # task Legate Core Toplevel Task (ID 1) that would ultimately result in
        # deadlock. Instead you receive this error message. Try unmapping the
        # region before invoking 'attach_external_resource'.
        runtime.issue_execution_fence(block=True)
        arr = np.random.rand(*shape)
        store = runtime.create_store_from_buffer(
            ty.float64, arr.shape, arr, False
        )
        partition = store.partition_by_tiling((2, 2, 2, 2))
        assert store.equal_storage(partition.store())
        child_store = partition.get_child_store(
            *[idx - 1 if idx > 0 else 0 for idx in partition.color_shape]
        )
        assert not child_store.equal_storage(store)
        child_store_arr = np.asarray(child_store.get_physical_store())
        arr_index = [idx % 2 - 2 for idx in shape]
        arr_p = arr[
            arr_index[0] :, arr_index[1] :, arr_index[2] :, arr_index[3] :
        ]
        np.testing.assert_allclose(child_store_arr, arr_p)

    @pytest.mark.parametrize(
        ("arr_shape", "dim", "index"),
        [
            ((2, 2), 1, 0),
            ((3, 2, 12), 2, 5),
            ((1024,), 0, 111),
            ((1, 2, 4, 8), 3, 7),
            (
                range(1, LEGATE_MAX_DIM + 1),
                LEGATE_MAX_DIM - 1,
                LEGATE_MAX_DIM // 2,
            ),
        ],
        ids=str,
    )
    def test_project(
        self, arr_shape: tuple[int, ...], dim: int, index: int
    ) -> None:
        arr, store = utils.random_array_and_store(arr_shape)
        projected_store = store.project(dim, index)
        projected_arr = np.asarray(projected_store.get_physical_store())
        np.testing.assert_allclose(arr.take(index, dim), projected_arr)

    def test_project_by_index(self) -> None:
        arr_shape = (7, 6, 5)
        arr, store = utils.random_array_and_store(arr_shape)
        projected_store = store[3]
        projected_arr = np.asarray(projected_store.get_physical_store())
        np.testing.assert_allclose(arr.take(3, 0), projected_arr)

    @pytest.mark.parametrize(
        ("shape", "dim"),
        [
            # TODO(yimoj) [issue 498]
            # crashes application when inline allocation is accessed
            # legion_python: /legion-src/runtime/legion/legion_domain.inl:954:
            # Legion::Domain::operator Legion::Rect<DIM, T>() const
            # [with int DIM = 1; T = long long int; Legion::Rect<DIM, T> =
            # Realm::Rect<1, long long int>]: Assertion `DIM == dim' failed.
            pytest.param((), 0, marks=pytest.mark.xfail(run=False)),
            ((1,), 0),
            ((1, 2, 3), 2),
            ((1024, 1, 1), 2),
            ((3, 4096), 1),
        ],
        ids=str,
    )
    @pytest.mark.parametrize("size", [0, 1, 252, 1024], ids=str)
    def test_promote(
        self, shape: tuple[int, ...], dim: int, size: int
    ) -> None:
        arr, store = utils.create_np_array_and_store(ty.int16, shape)
        expanded_arr = np.expand_dims(arr, dim)
        promoted = store.promote(dim, size)

        # TODO(yimoj) [issue 932]
        # when size > 0 np.asarray raises ValueError:
        # ValueError: mismatch in length of strides and shape
        promoted_arr = np.asarray(promoted.get_physical_store())
        assert promoted.equal_storage(store)
        assert np.allclose(expanded_arr, promoted_arr)

        # TODO(yimoj) [issue 932]
        # when size == 0 this fails due to the mismatching ndims:
        # E       assert 1 == 2
        # E        +  where 1 = array([257], dtype=int16).ndim
        # E        +  and   2 = array([], shape=(0, 1), dtype=int16).ndim
        assert expanded_arr.ndim == promoted_arr.ndim

        assert promoted_arr.shape[dim] == size

    def test_promote_by_index(self) -> None:
        arr, store = utils.create_np_array_and_store(ty.int16, (2, 5, 8))
        expanded_arr = np.expand_dims(arr, 0)
        promoted = store[(None,)]
        promoted_arr = np.asarray(promoted.get_physical_store())
        assert promoted.equal_storage(store)
        assert np.allclose(expanded_arr, promoted_arr)
        assert expanded_arr.ndim == promoted_arr.ndim
        assert promoted_arr.shape[0] == 1

    def test_numpy_ops(self) -> None:
        arr_shape = (1, 1, 2, 4)
        runtime = get_legate_runtime()
        arr = np.random.rand(*arr_shape)
        store = runtime.create_store_from_buffer(
            ty.float64, arr.shape, arr, False
        )
        store_arr = np.asarray(store.get_physical_store())
        store_arr.sort()
        # copy the store to double-check consistency between
        # store and interface
        new_store = runtime.create_store(ty.float64, shape=arr_shape)
        runtime.issue_copy(new_store, store)
        assert not new_store.equal_storage(store)
        new_arr = np.asarray(new_store.get_physical_store())
        np.testing.assert_allclose(new_arr, np.sort(arr))

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
    def test_transpose(
        self, arr_shape: tuple[int, ...], axes: tuple[int, ...]
    ) -> None:
        runtime = get_legate_runtime()
        arr = np.random.rand(*arr_shape)
        store = runtime.create_store_from_buffer(
            ty.float64, arr.shape, arr, False
        )
        arr_t = arr.transpose(axes)
        store_t = store.transpose(axes)
        store_arr = np.asarray(store_t.get_physical_store())
        np.testing.assert_allclose(store_arr, arr_t)

    @pytest.mark.parametrize(
        ("arr_shape", "dim", "start", "stop"),
        [
            ((2, 2), 1, None, 1),
            ((3, 2, 12), 2, 7, 5),
            ((1024,), 0, 111, 123),
            ((1024, 0, 1), 0, 0, None),
            ((1, 2, 4, 8), 3, 2, None),
            (
                range(1, LEGATE_MAX_DIM + 1),
                LEGATE_MAX_DIM - 1,
                LEGATE_MAX_DIM // 3,
                LEGATE_MAX_DIM // 2,
            ),
            pytest.param(
                range(LEGATE_MAX_DIM),
                LEGATE_MAX_DIM - 1,
                LEGATE_MAX_DIM // 3,
                LEGATE_MAX_DIM // 2,
                marks=pytest.mark.xfail(
                    # TODO(yimoj) [issue 522]
                    # LogicalStore Shape(0,1,2,1)
                    # __array_interface__ (0, 1, 2, 3)
                    reason="wrong shape in array interface"
                ),
            ),
        ],
        ids=str,
    )
    def test_slice(
        self,
        arr_shape: tuple[int, ...],
        dim: int,
        start: int | None,
        stop: int | None,
    ) -> None:
        arr, store = utils.random_array_and_store(arr_shape)
        slice_expr = (*(slice(None) for _ in range(dim)), slice(start, stop))
        expected = arr[slice_expr]
        sliced_store = store.slice(dim, slice(start, stop))
        sliced_arr = np.asarray(sliced_store.get_physical_store())
        np.testing.assert_allclose(sliced_arr, expected)
        np.testing.assert_allclose(
            np.asarray(store[slice_expr].get_physical_store()), sliced_arr
        )

    def test_slice_negative_dim(self) -> None:
        arr_shape = (1, 2, 3)
        arr, store = utils.random_array_and_store(arr_shape)
        slice_expr = (*(slice(None) for _ in range(2)), slice(None, 1))
        expected = arr[slice_expr]
        sliced_store = store.slice(-1, slice(None, 1))
        sliced_arr = np.asarray(sliced_store.get_physical_store())
        np.testing.assert_allclose(sliced_arr, expected)
        np.testing.assert_allclose(
            np.asarray(store[slice_expr].get_physical_store()), sliced_arr
        )

    def test_store_overlaps(self) -> None:
        runtime = get_legate_runtime()
        store = runtime.create_store(ty.int64, shape=(4, 4, 4))
        _, store = utils.random_array_and_store((5,))
        substore1 = store[:2]
        subarr1 = np.asarray(substore1.get_physical_store())
        substore2 = store[1:]
        subarr2 = np.asarray(substore2.get_physical_store())
        assert substore1.overlaps(substore2)
        assert len(np.intersect1d(subarr1, subarr2)) == 1

    def test_store_detatch(self) -> None:
        arr, store = utils.zero_array_and_store(ty.int32, (4, 1, 3))
        store.detach()
        store.fill(1)
        np.testing.assert_allclose(arr, np.zeros((4, 1, 3)))


class TestLogicalStoreOperationErrors:
    @pytest.mark.xfail(reason="LEGATE_MAX_DIM is not respected")
    def test_promote_exceed_max_dim(self) -> None:
        shape = range(1, LEGATE_MAX_DIM + 1)
        runtime = get_legate_runtime()
        store = runtime.create_store(ty.float64, shape)
        with pytest.raises(IndexError, match="maximum number of dimensions"):
            # TODO(yimoj) [issue 288]
            # promote does not respect LEGATE_MAX_DIM
            store.promote(0, 1)

    def test_promote_out_of_bounds_axis(self) -> None:
        runtime = get_legate_runtime()
        store = runtime.create_store(ty.float64, (1, 2))
        with pytest.raises(ValueError, match="Invalid promotion on dimension"):
            store.promote(4, 0)

    def test_delinearize_invalid_dim(self) -> None:
        runtime = get_legate_runtime()
        store = runtime.create_store(ty.float64, (1, 2, 3))
        with pytest.raises(ValueError, match="Invalid delinearization on dim"):
            store.delinearize(3, (1, 2))

    @pytest.mark.xfail(reason="LEGATE_MAX_DIM is not respected")
    def test_delinearize_exceed_max_dim(self) -> None:
        shape = (1, 2, 4, 5)
        runtime = get_legate_runtime()
        store = runtime.create_store(ty.int16, shape)
        with pytest.raises(IndexError, match="maximum number of dimensions"):
            store = store.delinearize(2, (1, 4))

        # we won't reach this line at this point, but keeping it here as it
        # crashes application when inline allocation is accessed since
        # max dim is exceeded:
        # Legion::DomainTransform::DomainTransform(const Legion::
        # DomainTransform&): Assertion `n <= LEGION_MAX_DIM' failed.

        # when max dim issue is fixed this assert should pass
        assert (
            store.get_physical_store().get_inline_allocation().shape == shape
        )

    def test_delinearize_invalid_shape(self) -> None:
        runtime = get_legate_runtime()
        store = runtime.create_store(ty.int16, (1,))
        with pytest.raises(
            ValueError, match=r"size.*cannot be delinearized into"
        ):
            store.delinearize(0, (1, 2))

    def test_project_invalid_dim(self) -> None:
        runtime = get_legate_runtime()
        store = runtime.create_store(ty.int16, shape=range(LEGATE_MAX_DIM))
        with pytest.raises(
            ValueError,
            match=r"Invalid projection on dimension .* for .* store",
        ):
            store.project(LEGATE_MAX_DIM, 0)

    def test_project_out_of_bounds_index(self) -> None:
        runtime = get_legate_runtime()
        store = runtime.create_store(ty.int16, shape=(1, 2, 3))
        msg = "Projection index .* is out of bounds"
        with pytest.raises(ValueError, match=msg):
            store.project(1, 5)

    @pytest.mark.parametrize("step", [-1, 2, 1.5, "1"])
    def test_unsupported_slice(self, step: Any) -> None:
        runtime = get_legate_runtime()
        store = runtime.create_store(ty.int16, (2, 2, 2))
        with pytest.raises(NotImplementedError, match="Unsupported slice"):
            store.slice(0, slice(0, None, step))

    def test_transpose_dimension_mismatch(self) -> None:
        runtime = get_legate_runtime()
        store = runtime.create_store(ty.int16, (2, 2, 2))
        with pytest.raises(ValueError, match="Dimension Mismatch"):
            store.transpose((0, 1))

    def test_transpose_invalid_axes(self) -> None:
        runtime = get_legate_runtime()
        store = runtime.create_store(ty.int16, (2, 2, 2))
        with pytest.raises(ValueError, match="Invalid axis"):
            store.transpose((1, 2, 3))

    def test_empty_volume(self) -> None:
        runtime = get_legate_runtime()
        store = runtime.create_store(ty.int16, (2, 2, 2))
        with pytest.raises(ValueError, match="volume greater than 0"):
            store.partition_by_tiling((0, 0, 0))

    @pytest.mark.parametrize(
        ("shape", "msg"),
        [
            ((1, 2), re.escape("Incompatible tile shape")),
            (1, re.escape("Expected an iterable but got <class 'int'>")),
        ],
    )
    def test_invalid_tile_shape(
        self, shape: tuple[int, ...] | int, msg: str
    ) -> None:
        runtime = get_legate_runtime()
        store = runtime.create_store(ty.int16, (2, 2, 2))
        with pytest.raises(ValueError, match=msg):
            store.partition_by_tiling(shape)  #  type: ignore[arg-type]

    def test_child_store_invalid_color_size(self) -> None:
        runtime = get_legate_runtime()
        store = runtime.create_store(ty.int16, (2, 2, 2))
        partition = store.partition_by_tiling((1, 1, 1))
        msg = re.escape(
            "Color [1] is invalid for partition of color shape [2, 2, 2]"
        )
        with pytest.raises(IndexError, match=msg):
            partition.get_child_store(1)

    def test_get_child_store_invalid_index(self) -> None:
        runtime = get_legate_runtime()
        store = runtime.create_store(ty.int16, (4, 6, 8))
        partition = store.partition_by_tiling((2, 2, 2))
        msg = "invalid for partition of color shape"
        with pytest.raises(IndexError, match=msg):
            partition.get_child_store(3, 4, 5)

    def test_partition_invalid_color(self) -> None:
        runtime = get_legate_runtime()
        arr = np.random.rand(*range(4))
        store = runtime.create_store_from_buffer(
            ty.float64, arr.shape, arr, False
        )
        partition = store.partition_by_tiling((2, 2, 2, 2))
        msg = "Color .* is invalid for partition of color shape"
        with pytest.raises(IndexError, match=msg):
            partition.get_child_store(
                *[idx - 1 if idx > 0 else 0 for idx in partition.color_shape]
            )

    def test_detach_no_attachment(self) -> None:
        store = get_legate_runtime().create_store(ty.int32, (1, 2, 3))
        with pytest.raises(
            ValueError, match="Store has no attachment to detach"
        ):
            store.detach()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
