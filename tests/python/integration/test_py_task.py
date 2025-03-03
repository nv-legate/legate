# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import numpy as np

import pytest

from legate.core import (
    ImageComputationHint,
    LogicalArray,
    Scalar,
    TaskTarget,
    Type,
    VariantCode,
    bloat,
    get_legate_runtime,
    image,
    types as ty,
)
from legate.core._lib.partitioning.constraint import scale
from legate.core.task import OutputStore, task

from .utils import tasks, utils
from .utils.data import ARRAY_TYPES, LARGE_SHAPES, SCALAR_VALS, SHAPES


class TestPyTask:
    @pytest.mark.parametrize("shape", SHAPES)
    def test_output_store(self, shape: tuple[int, ...]) -> None:
        arr, store = utils.random_array_and_store(shape)
        expected = np.zeros(arr.shape)
        tasks.zeros_task(store)
        np.testing.assert_allclose(arr, expected)

    @pytest.mark.parametrize("shape", SHAPES, ids=str)
    def test_input_output(self, shape: tuple[int, ...]) -> None:
        arr, store = utils.random_array_and_store(shape)
        out_arr, out_store = utils.empty_array_and_store(ty.float64, shape)
        tasks.copy_store_task(store, out_store)
        np.testing.assert_allclose(out_arr, arr)

    def test_mixed_input(self) -> None:
        arr1, store1 = utils.random_array_and_store((3, 2))
        arr2, store2 = utils.random_array_and_store((4, 3, 2))
        out_arr, out_store = utils.zero_array_and_store(
            ty.float64, (5, 4, 3, 2)
        )
        larr1 = LogicalArray.from_store(store1)
        exp = arr1 + arr2 + out_arr
        tasks.mixed_sum_task(larr1, store2, out_store)
        np.testing.assert_allclose(exp, out_arr)

    @pytest.mark.xfail(reason="LogicalStorePartition not supported")
    @pytest.mark.parametrize(
        "shape",
        [
            # LEGION ERROR: Invalid color space color for child 0 of logical
            # partition (1,1,1)
            pytest.param(
                (2, 4, 6, 8),
                marks=pytest.mark.xfail(
                    run=False, reason="crashes application"
                ),
            ),
            (3, 4, 6, 8),
            (7, 4, 3, 9),
            # LEGION ERROR: Invalid color space color for child 1 of
            # partition 3
            pytest.param(
                (500, 3, 3, 3),
                marks=pytest.mark.xfail(
                    run=False, reason="crashes application"
                ),
            ),
        ],
        ids=str,
    )
    def test_partition_to_store(self, shape: tuple[int, ...]) -> None:
        tile = (2, 2, 2, 2)
        arr, store = utils.random_array_and_store(shape)
        partition = store.partition_by_tiling(tile)
        out_arr, out_store = utils.zero_array_and_store(ty.float64, tile)
        # TypeError: Argument: 'partition' expected one of
        # (<class 'legate._lib.data.logical_store.LogicalStore'>,
        # <class 'legate._lib.data.logical_array.LogicalArray'>), got
        # <class 'legate._lib.data.logical_store.LogicalStorePartition'>
        tasks.partition_to_store_task(partition, out_store)
        np.testing.assert_allclose(out_arr, arr)

    def test_python_scalar_arg(self) -> None:
        @task(variants=tuple(VariantCode))
        def fill_task(val: int, out: OutputStore) -> None:
            out_arr = tasks.asarray(out.get_inline_allocation())
            out_arr.fill(val)

        val = 5
        out_arr, out_store = utils.empty_array_and_store(ty.int32, (1, 2, 3))
        fill_task(val, out_store)
        assert np.all(out_arr == val)

    @pytest.mark.parametrize(
        ("val", "dtype"), zip(SCALAR_VALS, ARRAY_TYPES), ids=str
    )
    def test_legate_scalar_arg(self, val: Any, dtype: ty.Type) -> None:
        runtime = get_legate_runtime()
        if (
            isinstance(val, bytes)
            and runtime.machine.preferred_target == TaskTarget.GPU
        ):
            pytest.skip("aborts proc with GPU")
        shape = (3, 1, 3)
        arr_np = np.full(shape, val, dtype=dtype.to_numpy_dtype())
        out_np, out_store = utils.empty_array_and_store(dtype, shape)
        tasks.fill_task(out_store, Scalar(val, dtype))
        if val is None or isinstance(val, bytes):
            assert arr_np.all() == out_np.all()
        else:
            np.testing.assert_allclose(arr_np, out_np)

    def test_scalar_arg(self) -> None:
        @task(variants=tuple(VariantCode))
        def fill_int_task(out: tasks.OutputArray, val: int) -> None:
            out_arr = tasks.asarray(out.data().get_inline_allocation())
            out_arr.fill(val)

        val = 12345
        shape = (3, 1, 3)
        dtype = ty.float16
        out_arr, out_store = utils.empty_array_and_store(dtype, shape)
        exp_arr = np.full(shape, val, dtype=dtype.to_numpy_dtype())

        fill_int_task(out_store, val)
        np.testing.assert_allclose(out_arr, exp_arr)

    @pytest.mark.parametrize("shape", SHAPES, ids=str)
    @pytest.mark.parametrize(
        ("dtype", "val"), zip(ARRAY_TYPES, SCALAR_VALS), ids=str
    )
    def test_ndarray_scalar_arg(
        self, shape: tuple[int, ...], dtype: Type, val: Any
    ) -> None:
        if val is None or (isinstance(val, bytes) and shape is None):
            pytest.skip(
                "numpy does not have a 0-sized type, so deducing the shape of "
                "the resulting store leads to size mismatches between the "
                "numpy type and legate type"
            )

        runtime = get_legate_runtime()
        if (
            isinstance(val, bytes)
            and runtime.machine.preferred_target == TaskTarget.GPU
        ):
            pytest.skip("aborts proc with GPU")
        out_arr, out_store = utils.empty_array_and_store(dtype, shape)
        in_arr = np.full(shape, val, dtype=dtype.to_numpy_dtype())
        tasks.copy_np_array_task(out_store, in_arr)
        runtime.issue_execution_fence(block=True)
        if val is None or isinstance(val, bytes):
            assert in_arr.all() == out_arr.all()
        else:
            np.testing.assert_allclose(in_arr, out_arr)

    @pytest.mark.parametrize("in_shape", SHAPES + LARGE_SHAPES, ids=str)
    def test_repeat_with_scale(self, in_shape: tuple[int, ...]) -> None:
        runtime = get_legate_runtime()
        in_np, in_store = utils.random_array_and_store(in_shape)
        if (
            runtime.machine.preferred_target == TaskTarget.GPU
            and in_np.size >= 1024
        ):
            pytest.xfail(reason="cupy reporting overflow")
        # Need to cast to int since randint() returns signedinteger[_32Bit |
        # _64Bit] since numpy 2.13
        repeats = tuple(map(int, np.random.randint(1, 3, in_np.ndim)))

        out_shape = tuple(s * r for s, r in zip(in_shape, repeats))
        out_np, out_store = utils.zero_array_and_store(ty.float64, out_shape)
        auto_task = tasks.repeat_task.prepare_call(
            in_store, out_store, repeats
        )

        in_arr = LogicalArray.from_store(in_store)
        in_part = auto_task.find_or_declare_partition(in_arr)
        out_arr = LogicalArray.from_store(out_store)
        out_part = auto_task.find_or_declare_partition(out_arr)
        auto_task.add_constraint(scale(tuple(repeats), in_part, out_part))
        auto_task.execute()
        runtime.issue_execution_fence(block=True)
        exp = in_np
        for i in range(in_np.ndim):
            exp = np.repeat(exp, repeats[i], axis=i)
        np.testing.assert_allclose(out_np, exp)

    def test_reduction(self) -> None:
        runtime = get_legate_runtime()
        in_arr = np.arange(10, dtype=np.float64)
        in_store = runtime.create_store_from_buffer(
            ty.float64, in_arr.shape, in_arr, False
        )
        out_arr, out_store = utils.zero_array_and_store(ty.float64, (0,))
        tasks.array_sum_task(in_store, out_store)
        np.testing.assert_allclose(out_arr, np.sum(in_arr))

    @pytest.mark.parametrize(
        "hint",
        (
            # TODO(wonchanl): currently, sparse sub-stores are not handled
            # correctly
            pytest.param(
                ImageComputationHint.NO_HINT,
                marks=pytest.mark.xfail(run=False),
            ),
            ImageComputationHint.MIN_MAX,
            ImageComputationHint.FIRST_LAST,
        ),
        ids=str,
    )
    def test_image_constraint_from_decorator(
        self, hint: ImageComputationHint
    ) -> None:
        py_task = task(
            variants=tuple(VariantCode),
            constraints=(image("func_store", "range_store", hint),),
        )(tasks.basic_image_task)

        runtime = get_legate_runtime()
        shape = (5, 4096, 5)
        func_shape = (2, 2048, 5)
        ndim = len(shape)
        indices = np.indices(shape)
        point_type = ty.point_type(ndim)
        points = np.stack(list(indices), axis=indices.ndim - 1).reshape(
            (indices.size // ndim, ndim)
        )

        func_arr = np.frombuffer(points, dtype=f"|V{point_type.size}").reshape(
            shape
        )
        rng = np.random.default_rng()
        rng.shuffle(func_arr)
        func_store = runtime.create_store_from_buffer(
            point_type, func_shape, func_arr[: np.prod(func_shape)], False
        )

        _, range_store = utils.random_array_and_store(shape)
        py_task(func_store, range_store)

    @pytest.mark.parametrize(
        "shape",
        [
            # TODO(yimoj): check and fix this later
            pytest.param((3, 2, 1), marks=pytest.mark.xfail(run=False)),
            pytest.param(
                (2, 1024, 1),
                # NotImplementedError: Unsupported type: (2, 1024, 1).
                # All elements must have the same type.
                # Element at index 1 has type uint16, expected uint8
                marks=pytest.mark.xfail(reason="insufficient dtype deduced"),
            ),
        ],
        ids=str,
    )
    def test_bloat_constraints(self, shape: tuple[int, ...]) -> None:
        low_offsets = tuple(np.random.randint(1, 6) for _ in shape)
        high_offsets = low_offsets[::-1]
        bloat_task = task(
            variants=tuple(VariantCode),
            constraints=(
                bloat("in_store", "bloat_store", low_offsets, high_offsets),
            ),
        )(tasks.basic_bloat_task)
        _, source_store = utils.random_array_and_store(shape)
        _, bloat_store = utils.random_array_and_store(shape)
        bloat_task(source_store, bloat_store, low_offsets, high_offsets, shape)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
