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

import re
from typing import Any, Type

import numpy as np
import pytest

from legate.core import (
    ImageComputationHint,
    LogicalArray,
    Scalar,
    Scope,
    TaskTarget,
    bloat,
    get_legate_runtime,
    image,
    scale,
    track_provenance,
    types as ty,
)
from legate.core.task import task

from .utils import tasks, utils
from .utils.data import (
    ARRAY_TYPES,
    EMPTY_SHAPES,
    LARGE_SHAPES,
    SCALAR_VALS,
    SHAPES,
)


class TestAutoTask:

    def test_create_auto_task(self) -> None:
        runtime = get_legate_runtime()
        library = runtime.core_library
        runtime.create_auto_task(library, tasks.basic_task.task_id)

    def test_default_provenance(self) -> None:
        runtime = get_legate_runtime()
        library = runtime.core_library
        auto_task = runtime.create_auto_task(library, tasks.basic_task.task_id)
        assert auto_task.provenance() == ""

    def test_scope_provenance(self) -> None:
        runtime = get_legate_runtime()
        library = runtime.core_library
        provenance = "foo"
        with Scope(provenance=provenance):
            auto_task = runtime.create_auto_task(
                library, tasks.basic_task.task_id
            )
        assert auto_task.provenance() == provenance

    def test_track_provenance(self) -> None:
        runtime = get_legate_runtime()
        library = runtime.core_library
        auto_task = track_provenance()(runtime.create_auto_task)(
            library, tasks.basic_task.task_id
        )
        pattern = r"[^/](/[^:]+):.*"
        found = re.search(pattern, auto_task.provenance())
        assert found
        assert found.groups()[0] == __file__

    @pytest.mark.parametrize("exc", [ValueError, AssertionError, RuntimeError])
    def test_pytask_exception_handling(self, exc: Type[Exception]) -> None:
        runtime = get_legate_runtime()
        library = runtime.core_library

        msg = "foo"

        @task
        def exc_func() -> None:
            raise exc(msg)

        auto_task = runtime.create_auto_task(library, exc_func.task_id)
        auto_task.throws_exception(exc)
        with pytest.raises(exc, match=msg):
            auto_task.execute()
        runtime.issue_execution_fence(block=True)

    def test_legate_exception_handling(self) -> None:
        runtime = get_legate_runtime()
        library = runtime.core_library
        auto_task = runtime.create_auto_task(library, tasks.basic_task.task_id)
        auto_task.add_output(runtime.create_store(ty.bool_))
        exc = ValueError
        auto_task.throws_exception(exc)
        with pytest.raises(exc, match="Wrong number of given arguments"):
            auto_task.execute()
        runtime.issue_execution_fence(block=True)

    def test_recreate_task_with_exception(self) -> None:
        runtime = get_legate_runtime()
        library = runtime.core_library
        exc = ValueError
        msg = "Wrong number of given arguments"

        def exc_task() -> None:
            exc_task = runtime.create_auto_task(
                library, tasks.basic_task.task_id
            )
            exc_task.add_output(runtime.create_store(ty.bool_))
            exc_task.throws_exception(exc)
            with pytest.raises(exc, match=msg):
                exc_task.execute()
            runtime.issue_execution_fence(block=True)

        for _ in range(3):
            exc_task()

    @pytest.mark.parametrize(
        "shape",
        [(1, 2, 1), (2, 101, 10), (3, 4096, 12), (65535, 1, 2)],
        ids=str,
    )
    def test_input_output(self, shape: tuple[int, ...]) -> None:
        runtime = get_legate_runtime()
        auto_task = runtime.create_auto_task(
            runtime.core_library, tasks.copy_store_task.task_id
        )

        in_arr_np = np.empty(shape=shape, dtype=np.int32)
        in_store = runtime.create_store_from_buffer(
            ty.int32, in_arr_np.shape, in_arr_np, False
        )
        out_store = runtime.create_store(ty.int32, shape)

        auto_task.add_input(in_store)
        auto_task.add_output(out_store)
        auto_task.add_alignment(out_store, in_store)
        auto_task.execute()
        runtime.issue_execution_fence(block=True)
        np.testing.assert_allclose(
            in_arr_np,
            np.asarray(out_store.get_physical_store().get_inline_allocation()),
        )

    @pytest.mark.parametrize(
        "val, dtype", zip(SCALAR_VALS, ARRAY_TYPES), ids=str
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
        auto_task = runtime.create_auto_task(
            runtime.core_library, tasks.fill_task.task_id
        )
        out_store = runtime.create_store(dtype, shape)
        auto_task.add_output(out_store)
        arr_np = np.full(shape, val, dtype=dtype_np)

        auto_task.add_scalar_arg(Scalar(val, dtype))
        auto_task.execute()
        runtime.issue_execution_fence(block=True)
        out_arr_np = np.asarray(
            out_store.get_physical_store().get_inline_allocation()
        )
        if val is None or isinstance(val, bytes):
            # if val is None, numpy complains that there is no operator-() for
            # NoneType for allclose. If val is bytes, then numpy complains that
            # it cannot be promoted to float. In either case, we can just
            # directly compare the objects.
            assert arr_np.all() == out_arr_np.all()
        else:
            np.testing.assert_allclose(arr_np, out_arr_np)

    @pytest.mark.parametrize("shape", SHAPES, ids=str)
    def test_tuple_scalar_arg(self, shape: tuple[int, ...]) -> None:
        runtime = get_legate_runtime()
        auto_task = runtime.create_auto_task(
            runtime.core_library, tasks.copy_np_array_task.task_id
        )
        out_store = runtime.create_store(ty.int32, shape)
        auto_task.add_output(out_store)

        arr_np = np.empty(shape=shape, dtype=np.int32)

        auto_task.add_scalar_arg(arr_np)
        auto_task.add_broadcast(out_store)
        auto_task.execute()
        runtime.issue_execution_fence(block=True)
        np.testing.assert_allclose(
            arr_np,
            np.asarray(out_store.get_physical_store().get_inline_allocation()),
        )

    def test_mixed_input(self) -> None:
        runtime = get_legate_runtime()
        auto_task = runtime.create_auto_task(
            runtime.core_library, tasks.mixed_sum_task.task_id
        )
        arg1_np = np.random.random((3, 2))
        arg2_np = np.random.random((4, 3, 2))
        out_np = np.zeros((5, 4, 3, 2))
        arg1_store = runtime.create_store_from_buffer(
            ty.float64, arg1_np.shape, arg1_np, False
        )
        arg2_store = runtime.create_store_from_buffer(
            ty.float64, arg2_np.shape, arg2_np, False
        )
        arg1_array = LogicalArray.from_store(arg1_store)
        out_store = runtime.create_store(ty.float64, (5, 4, 3, 2))
        auto_task.add_input(arg1_array)
        auto_task.add_input(arg2_store)
        auto_task.add_output(out_store)
        auto_task.add_broadcast(arg1_array)
        auto_task.add_broadcast(arg2_store)
        auto_task.add_broadcast(out_store)
        auto_task.execute()
        runtime.issue_execution_fence(block=True)
        np.testing.assert_allclose(
            arg1_np + arg2_np + out_np,
            np.asarray(out_store.get_physical_store().get_inline_allocation()),
        )

    @pytest.mark.parametrize(
        "initialize", [True, False], ids=["initialized", "uninitialized"]
    )
    @pytest.mark.parametrize("shape", SHAPES + EMPTY_SHAPES)
    def test_new_input_store(
        self, shape: tuple[int, ...], initialize: bool
    ) -> None:
        runtime = get_legate_runtime()
        auto_task = runtime.create_auto_task(
            runtime.core_library, tasks.copy_store_task.task_id
        )

        in_store = runtime.create_store(ty.int32, shape=shape)
        if initialize:
            in_store.fill(123)
        else:
            # TODO(yimoj) [issue 465]
            # this shouldn't be done, but it's allowed and works
            # so keeping it here instead of TestAutoTaskErrors for now
            in_store.get_physical_store().get_inline_allocation()

        out_store = runtime.create_store(ty.int32, shape)

        auto_task.add_input(in_store)
        auto_task.add_output(out_store)
        auto_task.add_alignment(out_store, in_store)
        auto_task.execute()
        runtime.issue_execution_fence(block=True)
        np.testing.assert_allclose(
            np.asarray(in_store.get_physical_store().get_inline_allocation()),
            np.asarray(out_store.get_physical_store().get_inline_allocation()),
        )

    @pytest.mark.parametrize("shape", SHAPES + EMPTY_SHAPES, ids=str)
    def test_concurrent(self, shape: tuple[int, ...]) -> None:
        runtime = get_legate_runtime()
        auto_task = runtime.create_auto_task(
            runtime.core_library, tasks.copy_store_task.task_id
        )
        in_arr, in_store = utils.random_array_and_store(shape)
        out_arr, out_store = utils.empty_array_and_store(ty.float64, shape)
        auto_task.add_input(in_store)
        auto_task.add_output(out_store)
        auto_task.add_alignment(out_store, in_store)
        # not sure if there's a way to confirm the effects from python side
        # just set val and execute for now
        auto_task.set_concurrent(True)
        auto_task.execute()
        runtime.issue_execution_fence(block=True)
        np.testing.assert_allclose(out_arr, in_arr)

    def test_side_effect(self) -> None:
        class foo:
            val = 0

        def increment() -> None:
            obj.val += 1

        @task(variants=tuple(tasks.KNOWN_VARIANTS))
        def foo_task() -> None:
            increment()

        obj = foo()
        runtime = get_legate_runtime()
        auto_task = runtime.create_auto_task(
            runtime.core_library, foo_task.task_id
        )
        # not sure how to actually check this from python side
        # just set and execute for now
        auto_task.set_side_effect(True)
        auto_task.execute()
        runtime.issue_execution_fence(block=True)
        assert obj.val == 1


class TestAutoTaskConstraints:

    @pytest.mark.xfail(
        get_legate_runtime().machine.count() > 1,
        run=False,
        reason="crashes on multi-node",
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

        auto_task = runtime.create_auto_task(
            runtime.core_library, tasks.array_sum_task.task_id
        )
        auto_task.add_input(in_store)
        auto_task.add_reduction(out_store, ty.ReductionOpKind.ADD)
        # TODO(yimoj) [issue 1099]
        # when running on multi-node this hits LEGION ERROR and then SIGABRT
        # LEGION ERROR: Error creating read-only field accessor without
        # read-only privileges on field 10000 in task
        auto_task.execute()
        runtime.issue_execution_fence(block=True)

        np.testing.assert_allclose(
            np.asarray(out_store.get_physical_store().get_inline_allocation()),
            np.sum(in_arr),
        )

    def test_add_broadcast(self) -> None:
        runtime = get_legate_runtime()
        count = runtime.machine.count()
        src_shape, tgt_shape = ((5, 1024, 5), (5, 1024 * count, 5))
        in_np, in_store = utils.random_array_and_store(src_shape)
        out_np, out_store = utils.zero_array_and_store(ty.float64, tgt_shape)

        auto_task = runtime.create_auto_task(
            runtime.core_library, tasks.copy_store_task.task_id
        )
        auto_task.add_input(in_store)
        auto_task.add_output(out_store)
        auto_task.add_broadcast(in_store)
        auto_task.execute()
        runtime.issue_execution_fence(block=True)

        for i in range(count):
            np.testing.assert_allclose(
                out_np[:, 1024 * i : 1024 * (i + 1), :], in_np
            )

    @pytest.mark.parametrize(
        "hint",
        (
            # TODO(wonchanl): currently, sparse sub-stores are not handled
            # correctly
            # ImageComputationHint.NO_HINT,
            ImageComputationHint.MIN_MAX,
            ImageComputationHint.FIRST_LAST,
        ),
        ids=str,
    )
    def test_image_constraint(self, hint: ImageComputationHint) -> None:
        image_task = task(variants=tuple(tasks.KNOWN_VARIANTS))(
            tasks.basic_image_task
        )

        runtime = get_legate_runtime()
        shape = (5, 4096, 5)
        func_shape = (2, 2048, 5)
        ndim = len(shape)
        indices = np.indices(shape)
        point_type = ty.point_type(ndim)
        points = np.stack([v for v in indices], axis=indices.ndim - 1).reshape(
            (indices.size // ndim, ndim)
        )

        func_arr = np.frombuffer(points, dtype=f"|V{point_type.size}").reshape(
            shape
        )
        rng = np.random.default_rng()
        rng.shuffle(func_arr)
        func_store = runtime.create_store_from_buffer(
            point_type,
            func_shape,
            func_arr[: np.prod(func_shape)],
            False,
        )

        _, range_store = utils.random_array_and_store(shape)
        auto_task = runtime.create_auto_task(
            runtime.core_library, image_task.task_id
        )
        func_part = auto_task.declare_partition()
        range_part = auto_task.declare_partition()
        auto_task.add_input(func_store, func_part)
        auto_task.add_input(range_store, range_part)
        auto_task.add_constraint(image(func_part, range_part, hint))
        auto_task.execute()

        runtime.issue_execution_fence(block=True)

    @pytest.mark.parametrize("in_shape", SHAPES + LARGE_SHAPES, ids=str)
    def test_repeat_with_scale(self, in_shape: tuple[int, ...]) -> None:
        runtime = get_legate_runtime()
        in_np, in_store = utils.random_array_and_store(in_shape)
        repeats = np.random.randint(1, 5, in_np.ndim)
        out_shape = tuple(
            in_shape[i] * repeats[i] for i in range(len(in_shape))
        )
        out_np, out_store = utils.zero_array_and_store(ty.float64, out_shape)

        auto_task = runtime.create_auto_task(
            runtime.core_library, tasks.repeat_task.task_id
        )
        in_part = auto_task.declare_partition()
        out_part = auto_task.declare_partition()
        auto_task.add_input(in_store, in_part)
        auto_task.add_output(out_store, out_part)
        auto_task.add_scalar_arg(repeats, (ty.int64,))
        auto_task.add_constraint(scale(tuple(repeats), in_part, out_part))
        auto_task.execute()
        runtime.issue_execution_fence(block=True)
        exp = in_np
        for i in range(in_np.ndim):
            exp = np.repeat(exp, repeats[i], axis=i)
        np.testing.assert_allclose(out_np, exp)

    @pytest.mark.parametrize("shape", SHAPES + LARGE_SHAPES, ids=str)
    def test_bloat_constraints(self, shape: tuple[int, ...]) -> None:

        bloat_task = task(variants=tuple(tasks.KNOWN_VARIANTS))(
            tasks.basic_bloat_task
        )

        low_offsets = tuple(np.random.randint(1, 6) for _ in shape)

        runtime = get_legate_runtime()
        high_offsets = low_offsets[::-1]
        _, source_store = utils.random_array_and_store(shape)
        _, bloat_store = utils.random_array_and_store(shape)

        auto_task = runtime.create_auto_task(
            runtime.core_library, bloat_task.task_id
        )
        source_part = auto_task.declare_partition()
        bloat_part = auto_task.declare_partition()
        auto_task.add_input(source_store, source_part)
        auto_task.add_input(bloat_store, bloat_part)
        auto_task.add_scalar_arg(low_offsets, (ty.int64,))
        auto_task.add_scalar_arg(high_offsets, (ty.int64,))
        auto_task.add_scalar_arg(shape, (ty.int64,))
        auto_task.add_constraint(
            bloat(source_part, bloat_part, low_offsets, high_offsets)
        )
        auto_task.execute()
        runtime.issue_execution_fence(block=True)


class TestAutoTaskErrors:
    def test_nonexistent_task_id(self) -> None:
        runtime = get_legate_runtime()
        msg = "does not have task"
        with pytest.raises(IndexError, match=msg):
            runtime.create_auto_task(runtime.core_library, -1)

    def test_add_invalid_input_output(self) -> None:
        runtime = get_legate_runtime()
        auto_task = runtime.create_auto_task(
            runtime.core_library, tasks.basic_task.task_id
        )
        msg = "Expected .* but got .*"
        with pytest.raises(ValueError, match=msg):
            auto_task.add_output("foo")  # type: ignore
        with pytest.raises(ValueError, match=msg):
            auto_task.add_input(Scalar(1, ty.int8))  # type: ignore

    @pytest.mark.parametrize(
        "dtype", [(ty.int32, ty.int64), (np.int32,)], ids=str
    )
    def test_unsupported_scalar_arg_type(self, dtype: tuple[Any, ...]) -> None:
        runtime = get_legate_runtime()
        auto_task = runtime.create_auto_task(
            runtime.core_library, tasks.basic_task.task_id
        )
        msg = "Unsupported type"
        with pytest.raises(TypeError, match=msg):
            auto_task.add_scalar_arg((123,), dtype)

    def test_scalar_val_with_array_type(self) -> None:
        runtime = get_legate_runtime()
        auto_task = runtime.create_auto_task(
            runtime.core_library, tasks.basic_task.task_id
        )
        msg = "object of type .* has no len()"
        with pytest.raises(TypeError, match=msg):
            auto_task.add_scalar_arg(123, (ty.int32,))

    @pytest.mark.xfail(run=False, reason="arbitrary crash during reuse")
    def test_auto_task_reuse(self) -> None:
        runtime = get_legate_runtime()
        library = runtime.core_library
        auto_task = runtime.create_auto_task(library, tasks.basic_task.task_id)
        auto_task.add_output(runtime.create_store(ty.bool_))
        auto_task.throws_exception(ValueError)
        msg = "Wrong number of given arguments"
        with pytest.raises(ValueError, match=msg):
            auto_task.execute()
        runtime.issue_execution_fence(block=True)
        # TODO(yimoj) [issue 384, 440]
        # reusing auto task should not be allowed
        # actual behavior to be updated after reuse check is implemented
        try:
            auto_task.execute()
        except Exception as exc:
            assert msg not in str(exc)
        runtime.issue_execution_fence(block=True)

    @pytest.mark.xfail(run=False, reason="crashes application")
    def test_uninitialized_input_store(self) -> None:
        runtime = get_legate_runtime()
        auto_task = runtime.create_auto_task(
            runtime.core_library, tasks.copy_store_task.task_id
        )

        in_store = runtime.create_store(ty.int32, shape=(1,))
        out_store = runtime.create_store(ty.int32, shape=(1,))

        auto_task.add_input(in_store)
        auto_task.add_output(out_store)
        auto_task.add_alignment(out_store, in_store)
        # TODO(yimoj) [issue 465]
        # crashes application if input store is not accessed prior
        # need to be updated when this is raising python exception properly
        auto_task.execute()

    def test_add_invalid_communicator(self) -> None:
        runtime = get_legate_runtime()
        library = runtime.core_library
        exc = RuntimeError
        msg = "No factory available for communicator"

        task = runtime.create_auto_task(library, tasks.basic_task.task_id)
        with pytest.raises(exc, match=msg):
            task.add_communicator("foo")

    @pytest.mark.parametrize("shape", LARGE_SHAPES, ids=str)
    def test_prefetched_store(self, shape: tuple[int, ...]) -> None:
        runtime = get_legate_runtime()
        dtype = ty.int32
        arr, store = utils.empty_array_and_store(dtype, shape)
        low_offsets = tuple(np.random.randint(1, 6) for _ in shape)
        high_offsets = low_offsets[::-1]
        runtime.prefetch_bloated_instances(
            store, low_offsets, high_offsets, True
        )
        auto_task = runtime.create_auto_task(
            runtime.core_library, tasks.fill_task.task_id
        )
        auto_task.add_output(store)
        val = 7654321
        exp = np.full(shape, val, dtype=dtype.to_numpy_dtype())
        auto_task.add_scalar_arg(Scalar(val, dtype))
        auto_task.execute()
        runtime.issue_execution_fence(block=True)
        # just check nothing is broken
        np.testing.assert_allclose(arr, exp)


class TestAutoTaskConstraintsErrors:

    def test_alignment_shape_mismatch(self) -> None:
        runtime = get_legate_runtime()
        auto_task = runtime.create_auto_task(
            runtime.core_library, tasks.copy_store_task.task_id
        )
        in_store = runtime.create_store(ty.int32, (1,))
        out_store = runtime.create_store(ty.int32, (1, 2))
        auto_task.add_input(in_store)
        auto_task.add_output(out_store)
        msg = "Alignment requires the stores to have the same shape"
        auto_task.add_alignment(out_store, in_store)
        with pytest.raises(ValueError, match=msg):
            auto_task.execute()
        runtime.issue_execution_fence(block=True)

    def test_alignment_bound_unbound(self) -> None:
        runtime = get_legate_runtime()
        auto_task = runtime.create_auto_task(
            runtime.core_library, tasks.copy_store_task.task_id
        )
        in_store = runtime.create_store(ty.int32, (1,))
        out_store = runtime.create_store(ty.int32)
        auto_task.add_input(in_store)
        auto_task.add_output(out_store)
        msg = "Alignment requires the stores to be all normal or all unbound"
        auto_task.add_alignment(out_store, in_store)
        with pytest.raises(ValueError, match=msg):
            auto_task.execute()
        runtime.issue_execution_fence(block=True)

    def test_alignment_non_input(self) -> None:
        runtime = get_legate_runtime()
        auto_task = runtime.create_auto_task(
            runtime.core_library, tasks.copy_store_task.task_id
        )
        in_store = runtime.create_store(ty.int32, (1,))
        tmp_store = runtime.create_store(ty.int32, (1,))
        auto_task.add_input(in_store)
        auto_task.add_alignment(in_store, tmp_store)
        # not asserting on the error message for this particular case due to
        # the message being unintuitive and may be compiler-dependent
        # match="unordered_map::at"
        with pytest.raises(IndexError):
            auto_task.execute()
        runtime.issue_execution_fence(block=True)

    @pytest.mark.parametrize("offsets", [(1, 2), (1, 2, 3, 4)], ids=str)
    def test_bloat_offset_mismatching_dims(
        self, offsets: tuple[int, ...]
    ) -> None:
        shape = (1, 2, 3)

        runtime = get_legate_runtime()
        _, source_store = utils.random_array_and_store(shape)
        _, bloat_store = utils.random_array_and_store(shape)

        auto_task = runtime.create_auto_task(
            runtime.core_library, tasks.copy_store_task.task_id
        )
        source_part = auto_task.declare_partition()
        bloat_part = auto_task.declare_partition()
        auto_task.add_input(source_store, source_part)
        auto_task.add_output(bloat_store, bloat_part)
        auto_task.add_constraint(
            bloat(source_part, bloat_part, offsets, offsets)
        )
        msg = "Bloating constraint requires the number of offsets to match"
        with pytest.raises(ValueError, match=msg):
            auto_task.execute()
        runtime.issue_execution_fence(block=True)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
