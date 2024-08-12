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

from typing import Any

import numpy as np
import pytest

from legate.core import Scalar, get_legate_runtime, types as ty
from legate.core.task import task

from .utils import tasks
from .utils.data import ARRAY_TYPES, EMPTY_SHAPES, SCALAR_VALS, SHAPES


class TestManualTask:
    @pytest.mark.parametrize("shape", SHAPES, ids=str)
    def test_create_manual_task(self, shape: tuple[int, ...]) -> None:
        runtime = get_legate_runtime()
        library = runtime.core_library
        runtime.create_manual_task(library, tasks.basic_task.task_id, shape)

    def test_create_with_lower_bounds(self) -> None:
        runtime = get_legate_runtime()
        library = runtime.core_library
        runtime.create_manual_task(
            library,
            tasks.basic_task.task_id,
            (1, 2, 3),
            (0, 1, 2),
        )

    @pytest.mark.parametrize("exc", [ValueError, AssertionError, RuntimeError])
    def test_pytask_exception_handling(self, exc: type[Exception]) -> None:
        runtime = get_legate_runtime()
        library = runtime.core_library

        msg = "foo"

        @task
        def exc_func() -> None:
            raise exc(msg)

        manual_task = runtime.create_manual_task(
            library, exc_func.task_id, (1,)
        )
        manual_task.throws_exception(exc)
        with pytest.raises(exc, match=msg):
            manual_task.execute()
        runtime.issue_execution_fence(block=True)

    def test_legate_exception_handling(self) -> None:
        runtime = get_legate_runtime()
        library = runtime.core_library
        manual_task = runtime.create_manual_task(
            library, tasks.basic_task.task_id, (1,)
        )
        manual_task.add_output(runtime.create_store(ty.bool_))
        exc = ValueError
        manual_task.throws_exception(exc)
        with pytest.raises(exc, match="Wrong number of given arguments"):
            manual_task.execute()
        runtime.issue_execution_fence(block=True)

    def test_recreate_task_with_exception(self) -> None:
        runtime = get_legate_runtime()
        library = runtime.core_library
        exc = ValueError
        msg = "Wrong number of given arguments"

        def exc_task() -> None:
            exc_task = runtime.create_manual_task(
                library, tasks.basic_task.task_id, (1,)
            )
            exc_task.add_output(runtime.create_store(ty.bool_))
            exc_task.throws_exception(exc)
            with pytest.raises(exc, match=msg):
                exc_task.execute()
            runtime.issue_execution_fence(block=True)

        for _ in range(3):
            exc_task()

    @pytest.mark.parametrize("shape", SHAPES, ids=str)
    def test_input_output(self, shape: tuple[int, ...]) -> None:
        runtime = get_legate_runtime()
        manual_task = runtime.create_manual_task(
            runtime.core_library, tasks.copy_store_task.task_id, shape
        )

        in_arr_np = np.ndarray(shape=shape, dtype=np.int32)
        in_store = runtime.create_store_from_buffer(
            ty.int32, in_arr_np.shape, in_arr_np, False
        )
        out_store = runtime.create_store(ty.int32, shape)

        manual_task.add_input(in_store)
        manual_task.add_output(out_store)
        manual_task.execute()
        runtime.issue_execution_fence(block=True)
        np.testing.assert_allclose(
            in_arr_np,
            np.asarray(out_store.get_physical_store().get_inline_allocation()),
        )

    @pytest.mark.parametrize(
        "val, dtype", zip(SCALAR_VALS, ARRAY_TYPES), ids=str
    )
    def test_scalar_arg(self, val: Any, dtype: ty.Type) -> None:
        shape = (3, 1, 3)
        dtype_np = dtype.to_numpy_dtype()
        runtime = get_legate_runtime()
        manual_task = runtime.create_manual_task(
            runtime.core_library, tasks.fill_task.task_id, shape
        )
        out_store = runtime.create_store(dtype, shape)
        manual_task.add_output(out_store)

        arr_np = np.full(shape, val, dtype=dtype_np)

        manual_task.add_scalar_arg(Scalar(val, dtype))
        manual_task.execute()
        runtime.issue_execution_fence(block=True)
        np.testing.assert_allclose(
            arr_np,
            np.asarray(out_store.get_physical_store().get_inline_allocation()),
        )

    @pytest.mark.parametrize("size", [9, 101], ids=str)
    def test_tuple_scalar_arg(self, size: int) -> None:
        shape = (size,)
        runtime = get_legate_runtime()
        manual_task = runtime.create_manual_task(
            runtime.core_library, tasks.copy_np_array_task.task_id, shape
        )
        out_store = runtime.create_store(ty.int32, shape)
        manual_task.add_output(out_store)

        arr_np = np.ndarray(shape=shape, dtype=np.int32)

        manual_task.add_scalar_arg(arr_np, (ty.int32,))
        manual_task.execute()
        runtime.issue_execution_fence(block=True)
        np.testing.assert_allclose(
            arr_np,
            np.asarray(out_store.get_physical_store().get_inline_allocation()),
        )

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
        runtime = get_legate_runtime()
        manual_task = runtime.create_manual_task(
            runtime.core_library, tasks.partition_to_store_task.task_id, tile
        )
        arr = np.random.random(shape)
        store = runtime.create_store_from_buffer(
            ty.float64, arr.shape, arr, False
        )
        partition = store.partition_by_tiling(tile)
        out_arr = np.zeros(tile)
        out_store = runtime.create_store_from_buffer(
            ty.float64, out_arr.shape, out_arr, False
        )
        manual_task.add_input(partition)
        manual_task.add_output(out_store)
        manual_task.execute()
        runtime.issue_execution_fence(block=True)

    def test_add_reduction(self) -> None:
        runtime = get_legate_runtime()
        in_arr = np.arange(10, dtype=np.float64)
        in_store = runtime.create_store_from_buffer(
            ty.float64, in_arr.shape, in_arr, False
        )
        out_arr = np.array((0,), dtype=np.float64)
        out_store = runtime.create_store_from_buffer(
            ty.float64, out_arr.shape, out_arr, False
        )

        manual_task = runtime.create_manual_task(
            runtime.core_library, tasks.array_sum_task.task_id, (1,)
        )

        manual_task.add_input(in_store)
        manual_task.add_reduction(out_store, ty.ReductionOpKind.ADD)
        manual_task.throws_exception(Exception)
        manual_task.execute()
        np.testing.assert_allclose(
            np.asarray(out_store.get_physical_store().get_inline_allocation()),
            np.sum(in_arr),
        )


class TestManualTaskErrors:
    def test_add_invalid_input_output(self) -> None:
        runtime = get_legate_runtime()
        manual_task = runtime.create_manual_task(
            runtime.core_library, tasks.basic_task.task_id, (1, 2, 3)
        )
        msg = "Expected .* but got .*"
        with pytest.raises(ValueError, match=msg):
            manual_task.add_output("foo")  # type: ignore
        with pytest.raises(ValueError, match=msg):
            manual_task.add_input(Scalar(1, ty.int8))  # type: ignore

    @pytest.mark.parametrize(
        "dtype", [(ty.int32, ty.int64), (np.int32,)], ids=str
    )
    def test_unsupported_scalar_arg_type(self, dtype: tuple[Any, ...]) -> None:
        runtime = get_legate_runtime()
        manual_task = runtime.create_manual_task(
            runtime.core_library, tasks.basic_task.task_id, (1, 2, 3)
        )
        msg = "Unsupported type"
        with pytest.raises(TypeError, match=msg):
            manual_task.add_scalar_arg((123,), dtype)

    def test_scalar_val_with_array_type(self) -> None:
        runtime = get_legate_runtime()
        manual_task = runtime.create_manual_task(
            runtime.core_library, tasks.basic_task.task_id, (1, 2, 3)
        )
        msg = "object of type .* has no len()"
        with pytest.raises(TypeError, match=msg):
            manual_task.add_scalar_arg(123, (ty.int32,))

    @pytest.mark.xfail(run=False, reason="crash during reuse")
    def test_manual_task_exception_reuse(self) -> None:
        runtime = get_legate_runtime()
        library = runtime.core_library
        manual_task = runtime.create_manual_task(
            library, tasks.basic_task.task_id, (1,)
        )
        manual_task.add_output(runtime.create_store(ty.bool_))
        manual_task.throws_exception(ValueError)
        msg = "Wrong number of given arguments"
        with pytest.raises(ValueError, match=msg):
            manual_task.execute()
        runtime.issue_execution_fence(block=True)
        # TODO(yimoj) [issue 384]
        # reusing task should not be allowed
        # actual behavior to be updated after reuse check is implemented
        try:
            manual_task.execute()
        except Exception as exc:
            assert msg not in str(exc)
        runtime.issue_execution_fence(block=True)

    @pytest.mark.parametrize(
        "shape, bounds",
        [(SHAPES[0], EMPTY_SHAPES[0]), (SHAPES[1], SHAPES[1])],
        ids=str,
    )
    def test_empty_domain(
        self, shape: tuple[int, ...], bounds: tuple[int, ...]
    ) -> None:
        runtime = get_legate_runtime()
        library = runtime.core_library
        with pytest.raises(ValueError, match="domain must not be empty"):
            runtime.create_manual_task(
                library, tasks.basic_task.task_id, shape, bounds
            )

    @pytest.mark.xfail(run=False, reason="crashes application")
    def test_launch_output_shape_mismatch(self) -> None:
        runtime = get_legate_runtime()
        library = runtime.core_library
        exc = ValueError
        msg = "Dimensionalities of output regions must be the same"

        exc_task = runtime.create_manual_task(
            library, tasks.basic_task.task_id, (1, 2, 3)
        )
        exc_task.add_output(runtime.create_store(ty.bool_))
        exc_task.throws_exception(exc)
        with pytest.raises(exc, match=msg):
            # TODO(yimoj) [issue 450]
            # Error same as issue 450, exception not catchable in python
            # [error 609] LEGION ERROR: Output region 0 of task basic_task
            # is requested to have 1 dimensions, but the color space has 3
            # dimensions. Dimensionalities of output regions must be the same
            # as the color space's in global indexing mode.
            exc_task.execute()
        runtime.issue_execution_fence(block=True)

    def test_create_empty_launch_domain(self) -> None:
        runtime = get_legate_runtime()
        msg = "Launch domain must not be empty"

        with pytest.raises(ValueError, match=msg):
            runtime.create_manual_task(
                runtime.core_library, tasks.basic_task.task_id, (0, 0, 0)
            )

    @pytest.mark.xfail(run=False, reason="crashes application")
    def test_uninitialized_input_store(self) -> None:
        runtime = get_legate_runtime()
        shape = (1,)
        manual_task = runtime.create_manual_task(
            runtime.core_library, tasks.copy_store_task.task_id, shape
        )

        in_store = runtime.create_store(ty.int32, shape=shape)
        out_store = runtime.create_store(ty.int32, shape)

        manual_task.add_input(in_store)
        manual_task.add_output(out_store)
        # TODO(yimoj) [issue 465]
        # crashes application if input store is not accessed prior
        # need to be updated when this is raising python exception properly
        manual_task.execute()

    def test_add_invalid_communicator(self) -> None:
        runtime = get_legate_runtime()
        library = runtime.core_library
        exc = RuntimeError
        msg = "No factory available for communicator"

        task = runtime.create_manual_task(
            library, tasks.basic_task.task_id, (1,)
        )
        with pytest.raises(exc, match=msg):
            task.add_communicator("foo")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
