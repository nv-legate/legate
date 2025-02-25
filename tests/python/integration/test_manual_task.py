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

from legate.core import (
    LEGATE_MAX_DIM,
    ManualTask,
    Scalar,
    Scope,
    TaskTarget,
    VariantCode,
    constant,
    get_legate_runtime,
    track_provenance,
    types as ty,
)
from legate.core.task import task

from .utils import tasks, utils
from .utils.data import ARRAY_TYPES, EMPTY_SHAPES, SCALAR_VALS, SHAPES


class TestManualTask:
    @pytest.mark.parametrize("shape", SHAPES, ids=str)
    def test_create_manual_task(self, shape: tuple[int, ...]) -> None:
        runtime = get_legate_runtime()
        library = runtime.core_library
        manual_task = runtime.create_manual_task(
            library, tasks.basic_task.task_id, shape
        )
        assert isinstance(manual_task, ManualTask)
        # just touching raw_handle for coverage
        _ = manual_task.raw_handle

    def test_create_with_lower_bounds(self) -> None:
        runtime = get_legate_runtime()
        library = runtime.core_library
        runtime.create_manual_task(
            library, tasks.basic_task.task_id, (1, 2, 3), (0, 1, 2)
        )

    def test_default_provenance(self) -> None:
        runtime = get_legate_runtime()
        library = runtime.core_library
        manual_task = runtime.create_manual_task(
            library, tasks.basic_task.task_id, (1, 2, 3)
        )
        assert manual_task.provenance() == ""

    def test_scope_provenance(self) -> None:
        runtime = get_legate_runtime()
        library = runtime.core_library
        provenance = "foo"
        with Scope(provenance=provenance):
            manual_task = runtime.create_manual_task(
                library, tasks.basic_task.task_id, (1, 2, 3)
            )
        assert manual_task.provenance() == provenance

    def test_track_provenance(self) -> None:
        runtime = get_legate_runtime()
        library = runtime.core_library
        manual_task = track_provenance()(runtime.create_manual_task)(
            library, tasks.basic_task.task_id, (1, 2, 3)
        )
        pattern = r"[^/](/[^:]+):.*"
        match = re.search(pattern, manual_task.provenance())
        assert match
        assert match.groups()[0] == __file__

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
        exc = IndexError
        manual_task.throws_exception(exc)
        with pytest.raises(
            exc,
            match=re.escape(
                "Invalid arguments to task. Expected Nargs(0) output "
                "arguments, have 1"
            ),
        ):
            manual_task.execute()
        runtime.issue_execution_fence(block=True)

    def test_recreate_task_with_exception(self) -> None:
        runtime = get_legate_runtime()
        library = runtime.core_library
        exc = IndexError
        msg = re.escape(
            "Invalid arguments to task. Expected Nargs(0) output "
            "arguments, have 1"
        )

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

        in_arr_np = np.empty(shape=shape, dtype=np.int32)
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
        ("val", "dtype"), zip(SCALAR_VALS, ARRAY_TYPES), ids=str
    )
    def test_scalar_arg(self, val: Any, dtype: ty.Type) -> None:
        runtime = get_legate_runtime()
        if (
            isinstance(val, bytes)
            and runtime.machine.preferred_target == TaskTarget.GPU
        ):
            pytest.skip("aborts proc with GPU")
        shape = (3, 1, 3)
        dtype_np = dtype.to_numpy_dtype()
        manual_task = runtime.create_manual_task(
            runtime.core_library, tasks.fill_task.task_id, shape
        )
        out_store = runtime.create_store(dtype, shape)
        manual_task.add_output(out_store)

        arr_np = np.full(shape, val, dtype=dtype_np)

        manual_task.add_scalar_arg(Scalar(val, dtype))
        manual_task.execute()
        runtime.issue_execution_fence(block=True)
        out_arr_np = np.asarray(out_store.get_physical_store())
        if val is None or isinstance(val, bytes):
            assert arr_np.all() == out_arr_np.all()
        else:
            np.testing.assert_allclose(arr_np, out_arr_np)

    @pytest.mark.parametrize("size", [9, 101], ids=str)
    def test_tuple_scalar_arg(self, size: int) -> None:
        shape = (size,)
        runtime = get_legate_runtime()
        manual_task = runtime.create_manual_task(
            runtime.core_library, tasks.copy_np_array_task.task_id, shape
        )
        out_store = runtime.create_store(ty.int32, shape)
        manual_task.add_output(out_store)

        arr_np = np.empty(shape=shape, dtype=np.int32)

        manual_task.add_scalar_arg(arr_np)
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

    @pytest.mark.parametrize("shape", [(3, 4, 6, 8), (7, 4, 3, 9)], ids=str)
    def test_copy_partition(self, shape: tuple[int, ...]) -> None:
        tile = (2, 2, 2, 2)
        runtime = get_legate_runtime()
        manual_task = runtime.create_manual_task(
            runtime.core_library, tasks.copy_store_task.task_id, tile
        )
        arr = np.random.random(shape)
        store = runtime.create_store_from_buffer(
            ty.float64, arr.shape, arr, False
        )
        out_arr = np.zeros(shape, dtype=np.float64)
        out_store = runtime.create_store_from_buffer(
            ty.float64, out_arr.shape, out_arr, False
        )
        in_partition = store.partition_by_tiling(tile)
        out_partition = out_store.partition_by_tiling(tile)
        manual_task.add_input(in_partition)
        manual_task.add_output(out_partition)
        manual_task.execute()
        runtime.issue_execution_fence(block=True)
        np.testing.assert_allclose(
            out_arr[:2, :2, :2, :2], arr[:2, :2, :2, :2]
        )

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
        manual_task.execute()
        np.testing.assert_allclose(
            np.asarray(out_store.get_physical_store().get_inline_allocation()),
            np.sum(in_arr),
        )

    def test_add_reduction_partition(self) -> None:
        runtime = get_legate_runtime()
        in_arr = np.arange(10, dtype=np.float64)
        in_store = runtime.create_store_from_buffer(
            ty.float64, in_arr.shape, in_arr, read_only=False
        )
        out_arr = np.zeros(shape=(2, 2), dtype=np.float64)
        out_store = runtime.create_store_from_buffer(
            ty.float64, out_arr.shape, out_arr, read_only=False
        )
        partition = out_store.partition_by_tiling((1, 1))

        manual_task = runtime.create_manual_task(
            runtime.core_library, tasks.array_sum_task.task_id, (1,)
        )

        manual_task.add_input(in_store)
        manual_task.add_reduction(
            partition, ty.ReductionOpKind.ADD, (constant(1), constant(0))
        )
        manual_task.execute()
        runtime.issue_execution_fence(block=True)
        np.testing.assert_allclose(out_arr[1, 0], np.sum(in_arr))

    def test_concurrent(self) -> None:
        runtime = get_legate_runtime()
        shape = (runtime.machine.count(),)
        manual_task = runtime.create_manual_task(
            runtime.core_library, tasks.copy_store_task.task_id, shape
        )
        in_arr, in_store = utils.random_array_and_store(shape)
        out_arr, out_store = utils.empty_array_and_store(ty.float64, shape)
        manual_task.add_input(in_store)
        manual_task.add_output(out_store)
        # not sure if there's a way to confirm the effects from python side
        # just set val and execute for now
        manual_task.set_concurrent(True)
        manual_task.execute()
        runtime.issue_execution_fence(block=True)
        np.testing.assert_allclose(out_arr, in_arr)

    def test_side_effect(self) -> None:
        class foo:
            val = 0

        def increment() -> None:
            obj.val += 1

        @task(variants=tuple(VariantCode))
        def foo_task() -> None:
            increment()

        obj = foo()
        runtime = get_legate_runtime()
        count = runtime.machine.count()
        manual_task = runtime.create_manual_task(
            runtime.core_library, foo_task.task_id, (count,)
        )
        # not sure how to actually check the impact from python side
        # just set and execute for now
        manual_task.set_side_effect(True)
        manual_task.execute()
        runtime.issue_execution_fence(block=True)
        assert obj.val == count

    @pytest.mark.parametrize("communicator", ["cpu", "nccl", "cal"])
    def test_add_communicator(self, communicator: str) -> None:
        runtime = get_legate_runtime()
        manual_task = runtime.create_manual_task(
            runtime.core_library, tasks.basic_task.task_id, (1,)
        )
        # can't find a good way to check whether a communicator exists or not
        # before the acture add_communicator call
        # just check that they either exist or raise the expected msg
        expected = f"No factory available for communicator '{communicator}'"
        raised = None
        try:
            manual_task.add_communicator(communicator)
        except RuntimeError as exc:
            raised = exc.args[0]
        # "cpu" should always exist
        if communicator == "cpu":
            assert raised is None
        elif raised:
            assert expected in raised
        manual_task.execute()

    @pytest.mark.parametrize("communicator", ["cpu", "nccl", "cal"])
    def test_builtin_communicator(self, communicator: str) -> None:
        runtime = get_legate_runtime()
        manual_task = runtime.create_manual_task(
            runtime.core_library,
            tasks.basic_task.task_id,
            (1,) * min(runtime.machine.count(), LEGATE_MAX_DIM),
        )

        func = getattr(manual_task, f"add_{communicator}_communicator")
        expected = f"No factory available for communicator '{communicator}'"
        raised = None
        try:
            func()
        except RuntimeError as exc:
            raised = exc.args[0]
        # "cpu" should always exist
        if communicator == "cpu":
            assert raised is None
        elif raised:
            assert expected in raised
        manual_task.execute()


class TestManualTaskErrors:
    def test_add_invalid_input_output(self) -> None:
        runtime = get_legate_runtime()
        manual_task = runtime.create_manual_task(
            runtime.core_library, tasks.basic_task.task_id, (1, 2, 3)
        )
        msg = "Expected .* but got .*"
        with pytest.raises(TypeError, match=msg):
            manual_task.add_output("foo")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match=msg):
            manual_task.add_input(Scalar(1, ty.int8))  # type: ignore[arg-type]

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
            assert msg not in str(exc)  # noqa: PT017
        runtime.issue_execution_fence(block=True)

    @pytest.mark.parametrize(
        ("shape", "bounds"),
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

    @pytest.mark.parametrize(
        ("launch_domain", "msg"),
        [
            ((0, 0, 0), "Launch domain must not be empty"),
            (1, "Launch space must be iterable"),
        ],
    )
    def test_create_invalid_launch_domain(
        self, launch_domain: Any, msg: str
    ) -> None:
        runtime = get_legate_runtime()
        with pytest.raises(ValueError, match=msg):
            runtime.create_manual_task(
                runtime.core_library, tasks.basic_task.task_id, launch_domain
            )

    def test_create_invalid_lower_bounds(self) -> None:
        runtime = get_legate_runtime()
        msg = "Lower bounds must be iterable"
        with pytest.raises(ValueError, match=msg):
            runtime.create_manual_task(
                runtime.core_library,
                tasks.basic_task.task_id,
                (1,),
                1,  # type:ignore [arg-type]
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

    @pytest.mark.xfail(run=False, reason="crashes application")
    def test_invalid_concurrent_mapping(self) -> None:
        runtime = get_legate_runtime()
        shape = (runtime.machine.count() + 1,)
        manual_task = runtime.create_manual_task(
            runtime.core_library, tasks.basic_task.task_id, shape
        )
        manual_task.set_concurrent(True)
        # TODO(yimoj) [issue 1261]
        # [error 67] LEGION ERROR: Mapper legate.core on Node 0 performed
        # illegal mapping of concurrent index space task foo (UID 5) by mapping
        # multiple points to the same processor 1d00000000000002. All point
        # tasks must be mapped to different processors for concurrent execution
        # of index space tasks.
        manual_task.execute()
        runtime.issue_execution_fence(block=True)

    def test_invalid_projection_type(self) -> None:
        runtime = get_legate_runtime()
        out_arr = np.array((2, 2), dtype=np.float64)
        out_store = runtime.create_store_from_buffer(
            ty.float64, out_arr.shape, out_arr, False
        )
        partition = out_store.partition_by_tiling((1,))
        manual_task = runtime.create_manual_task(
            runtime.core_library, tasks.array_sum_task.task_id, (1,)
        )

        proj = "foo"
        msg = f"Expected a tuple, but got {type(proj)}"
        with pytest.raises(ValueError, match=msg):
            manual_task.add_reduction(
                partition,
                ty.ReductionOpKind.ADD,
                proj,  # type: ignore[arg-type]
            )

    def test_invalid_reduction_store(self) -> None:
        runtime = get_legate_runtime()
        manual_task = runtime.create_manual_task(
            runtime.core_library, tasks.array_sum_task.task_id, (1,)
        )

        store = "foo"
        msg = (
            "Expected a logical store or store partition "
            f"but got {type(store)}"
        )
        with pytest.raises(ValueError, match=msg):
            manual_task.add_reduction(
                store,  # type: ignore[arg-type]
                ty.ReductionOpKind.ADD,
            )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
