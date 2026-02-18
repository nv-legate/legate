# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import re
import sys
from typing import TYPE_CHECKING, Any

import numpy as np

try:
    import cupy  # type: ignore[import-not-found]
except ModuleNotFoundError:
    cupy = None

if TYPE_CHECKING:
    from collections.abc import Callable
    from subprocess import CompletedProcess

    from numpy.typing import NDArray

import pytest

from legate.core import (
    ImageComputationHint,
    LogicalArray,
    ParallelPolicy,
    Scalar,
    Scope,
    StreamingMode,
    TaskConfig,
    TaskTarget,
    Type,
    VariantCode,
    VariantOptions,
    align,
    bloat,
    broadcast,
    from_dlpack,
    get_legate_runtime,
    image,
    scale,
    types as ty,
)
from legate.core.task import (
    ADD,
    InputArray,
    InputStore,
    OutputStore,
    PyTask,
    ReductionStore,
    task,
)

from .utils import tasks, utils
from .utils.data import ARRAY_TYPES, LARGE_SHAPES, SCALAR_VALS, SHAPES
from .utils.utils import is_multi_node


class TestPyTask:
    @pytest.mark.parametrize("shape", SHAPES, ids=str)
    def test_task_from_object(self, shape: tuple[int, ...]) -> None:
        class TaskMeta(type):
            def __getattribute__(cls, name: str) -> Any:
                if name == "__qualname__":
                    raise AttributeError
                return super().__getattribute__(name)

        class TaskClass(metaclass=TaskMeta):
            def __getattribute__(self, name: str) -> Any:
                if name == "__qualname__":
                    raise AttributeError
                if name == "__name__":
                    return "TaskClass"
                return super().__getattribute__(name)

            def __call__(
                self, in_store: InputStore, out_store: OutputStore
            ) -> None:
                tasks._copy_store_task(in_store, out_store)

        arr, store = utils.random_array_and_store(shape)
        out_arr, out_store = utils.empty_array_and_store(ty.float64, shape)
        py_task = PyTask(func=TaskClass(), variants=tuple(VariantCode))
        py_task(store, out_store)
        np.testing.assert_allclose(out_arr, arr)

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

    @pytest.mark.parametrize("shape", SHAPES, ids=str)
    def test_dlpack_input_output(self, shape: tuple[int, ...]) -> None:
        arr = np.random.rand(*shape)
        out_arr = np.empty(shape)
        store = from_dlpack(arr)
        out_store = from_dlpack(utils.UnversionedDLPack(out_arr))
        tasks.copy_store_task(store, out_store)
        np.testing.assert_allclose(out_arr, arr)

    @pytest.mark.skipif(
        get_legate_runtime().machine.preferred_target != TaskTarget.GPU,
        reason="GPU only test",
    )
    @pytest.mark.parametrize("shape", SHAPES, ids=str)
    def test_cupy_dlpack_input_output(self, shape: tuple[int, ...]) -> None:
        if not cupy:
            pytest.skip(reason="test requires cupy")
        cupy.random.seed(42)
        arr = cupy.random.rand(*shape)
        out_arr = cupy.empty(shape)
        store = from_dlpack(arr)
        out_store = from_dlpack(utils.UnversionedDLPack(out_arr))
        tasks.copy_store_task(store, out_store)
        get_legate_runtime().issue_execution_fence(block=True)
        cupy.testing.assert_allclose(out_arr, arr)

    def test_fill_dlpack(self) -> None:
        store = get_legate_runtime().create_store(ty.int32, (3, 1, 3))
        tasks.fill_dlpack_task(store, 1)
        assert (np.asarray(store.get_physical_store()) == 1).all()

    @pytest.mark.skipif(
        TaskTarget.GPU not in get_legate_runtime().machine.valid_targets,
        reason="GPU only test",
    )
    @pytest.mark.parametrize("stream", [-1, 0, 2])
    def test_dlpack_stream(self, stream: int) -> None:
        @task(variants=(VariantCode.GPU,))
        def stream_task(store: InputStore) -> None:
            store.__dlpack__(stream=stream)

        store = get_legate_runtime().create_store(ty.int32, (3, 1, 3))
        store.fill(0)
        stream_task(store)

    @pytest.mark.parametrize("shape", SHAPES, ids=str)
    def test_input_output_prepare(self, shape: tuple[int, ...]) -> None:
        arr, store = utils.random_array_and_store(shape)
        out_arr, out_store = utils.empty_array_and_store(ty.float64, shape)
        auto_task = tasks.copy_store_task.prepare_call(store, out_store)
        auto_task.execute()
        get_legate_runtime().issue_execution_fence(block=True)
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
        ("val", "dtype"), zip(SCALAR_VALS, ARRAY_TYPES, strict=True), ids=str
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

    @pytest.mark.parametrize(
        "shape",
        [pytest.param((), marks=pytest.mark.xfail(run=False)), *SHAPES],
    )
    def test_bind_buffer(self, shape: tuple[int, ...]) -> None:
        val = 98765

        @task(
            variants=tuple(VariantCode),
            options=VariantOptions(has_allocations=True),
        )
        def bind(out: tasks.OutputStore) -> None:
            buf = out.create_output_buffer(shape)
            tasks.asarray(buf).fill(val)

        runtime = get_legate_runtime()
        out_store = runtime.create_store(ty.int32, ndim=len(shape))
        exp_arr: NDArray[Any] = np.ndarray(shape, dtype=np.int32)
        exp_arr.fill(val)
        assert out_store.unbound
        # TODO(yimoj) [issue-2529]
        # Executing this task will hit Legion error when ndim == 0
        # Legion::OutputRequirement::OutputRequirement(Legion::FieldSpace,
        # const std::set<unsigned int>&, int, bool): Assertion `false' failed.
        bind(out_store)
        runtime.issue_execution_fence(block=True)
        assert not out_store.unbound
        np.testing.assert_allclose(
            np.asarray(out_store.get_physical_store()), exp_arr
        )

    @pytest.mark.parametrize(
        "shape",
        [pytest.param((), marks=pytest.mark.xfail(run=False)), *SHAPES],
    )
    def test_bind_deferred(self, shape: tuple[int, ...]) -> None:
        val1 = 98765
        shape1 = (1,) * len(shape)
        val2 = 54321
        shape2 = shape

        @task(
            variants=tuple(VariantCode),
            options=VariantOptions(has_allocations=True),
        )
        def mix_bind(out1: tasks.OutputStore, out2: tasks.OutputStore) -> None:
            buf1 = out2.create_output_buffer(shape, bind=False)
            tasks.asarray(buf1).fill(val1)
            buf2 = out1.create_output_buffer(shape, bind=False)
            tasks.asarray(buf2).fill(val2)
            out1.bind_data(buf1, shape1)
            out2.bind_data(buf2, shape2)

        runtime = get_legate_runtime()
        store1 = runtime.create_store(ty.int32, ndim=len(shape))
        store2 = runtime.create_store(ty.int32, ndim=len(shape))
        mix_bind(store1, store2)
        runtime.issue_execution_fence(block=True)
        assert store1.shape == shape1
        assert (np.asarray(store1) == val1).all()
        assert store2.shape == shape2
        assert (np.asarray(store2) == val2).all()

    @pytest.mark.xfail(run=False, reason="aborts python")
    def test_allocate_without_binding(self) -> None:
        @task(
            variants=tuple(VariantCode),
            options=VariantOptions(has_allocations=True),
        )
        def forgot_binding(out: tasks.OutputStore) -> None:
            buf = out.create_output_buffer((3, 2, 1), bind=False)
            tasks.asarray(buf).fill(1)

        runtime = get_legate_runtime()
        store = runtime.create_store(ty.int32, ndim=3)
        # TODO(yimoj)
        # this should be a pytest.raises when properly handled on python side
        # {legion}: LEGION PROGRAMMING MODEL EXCEPTION:
        # did not return any instance for field 10000of output requirement 0
        forgot_binding(store)
        runtime.issue_execution_fence(block=True)

    def test_binding_type_mismatch(self) -> None:
        @task(
            variants=tuple(VariantCode),
            options=VariantOptions(has_allocations=True),
        )
        def bad_binding(
            out1: tasks.OutputStore, out2: tasks.OutputStore
        ) -> None:
            buf1 = out1.create_output_buffer((3, 2, 1), bind=False)
            buf2 = out2.create_output_buffer((3, 2, 1), bind=False)
            msg = (
                f"Cannot bind data of type {buf1.type} to store of type "
                f"{buf2.type}, types are not compatible"
            )
            with pytest.raises(TypeError, match=msg):
                out2.bind_data(buf1)
            # still need to bind these to avoid
            # LEGION PROGRAMMING MODEL EXCEPTION
            out1.bind_data(buf1)
            out2.bind_data(buf2)

        runtime = get_legate_runtime()
        store1 = runtime.create_store(ty.int32, ndim=3)
        store2 = runtime.create_store(ty.float64, ndim=3)
        bad_binding(store1, store2)
        runtime.issue_execution_fence(block=True)

    @pytest.mark.xfail(run=False, reason="aborts python")
    def test_binding_without_allocation(self) -> None:
        @task(variants=tuple(VariantCode))
        def forgot_allocation(out: tasks.OutputStore) -> None:
            buf = out.create_output_buffer((3, 2, 1), bind=False)
            tasks.asarray(buf).fill(1)

        runtime = get_legate_runtime()
        store = runtime.create_store(ty.int32, ndim=3)
        # TODO(yimoj)
        # this should be a pytest.raises when properly handled on python side
        # {legion}: LEGION RESOURCE EXCEPTION:
        # Failed to allocate DeferredBuffer/Value/Reduction of 24 bytes for
        # leaf task TestPyTask.test_binding_without_allocation.<locals>.
        # bad_binding(UID: 3) in GPU_FB_MEM memory because there was
        # insufficient space reserved for dynamic allocations. Only 0 bytes
        # remain of 0 reserved bytes. This means that you set your upper bound
        # for the amount of dynamic memory required for this task too low.
        forgot_allocation(store)
        runtime.issue_execution_fence(block=True)

    @pytest.mark.xfail(run=False, reason="aborts python")
    def test_rebind(self) -> None:
        @task(
            variants=tuple(VariantCode),
            options=VariantOptions(has_allocations=True),
        )
        def bad_binding(
            out1: tasks.OutputStore, out2: tasks.OutputStore
        ) -> None:
            buf1 = out1.create_output_buffer((3, 2, 1), bind=False)
            # TODO(yimoj) this should be a pytest.raises when properly handled
            # on python side
            # const Realm::InstanceLayoutGeneric*
            # Realm::RegionInstance::get_layout() const: Assertion
            # `r_impl->metadata.is_valid() && "instance metadata must be valid
            # before accesses are performed"' failed.
            out1.bind_data(buf1)
            out2.bind_data(buf1)

        runtime = get_legate_runtime()
        store1 = runtime.create_store(ty.int32, ndim=3)
        store2 = runtime.create_store(ty.int32, ndim=3)
        bad_binding(store1, store2)
        runtime.issue_execution_fence(block=True)

    @pytest.mark.parametrize("shape", SHAPES, ids=str)
    @pytest.mark.parametrize(
        ("dtype", "val"), zip(ARRAY_TYPES, SCALAR_VALS, strict=True), ids=str
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

        out_shape = tuple(
            s * r for s, r in zip(in_shape, repeats, strict=True)
        )
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
        runtime.issue_execution_fence(block=True)
        np.testing.assert_allclose(out_arr, np.sum(in_arr))

    def test_reduction_constraint(self) -> None:
        runtime = get_legate_runtime()
        in_arr = np.arange(10, dtype=np.float64)
        in_store = runtime.create_store_from_buffer(
            ty.float64, in_arr.shape, in_arr, False
        )

        @task(variants=tuple(VariantCode), constraints=(broadcast("out"),))
        def array_sum_task(
            store: InputStore, out: ReductionStore[ADD]
        ) -> None:
            out_arr = tasks.asarray(out.get_inline_allocation())
            store_arr = tasks.asarray(store.get_inline_allocation())
            out_arr[:] = store_arr.sum()

        out_arr, out_store = utils.zero_array_and_store(ty.float64, (10,))
        array_sum_task(in_store, out_store)
        runtime.issue_execution_fence(block=True)
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

    @pytest.mark.skipif(
        # TODO(yimoj) [PR-3069]
        # Test fails in CI for some reason and the output is truncated.
        # Disable it in CI for now.
        bool(os.environ.get("CI"))
        or get_legate_runtime().machine.preferred_target != TaskTarget.CPU,
        reason="CPU only test",
    )
    def test_throws_exception(
        self, run_subprocess: Callable[..., CompletedProcess[Any]] | None
    ) -> None:
        if run_subprocess:
            run_subprocess(
                __file__,
                "TestPyTask::test_throws_exception",
                {
                    "LEGATE_MAX_EXCEPTION_SIZE": "15000",
                    "LEGATE_AUTO_CONFIG": "0",
                    "LEGATE_CONFIG": "--cpus 1",
                },
            )
            return

        @task(
            variants=tuple(VariantCode),
            options=VariantOptions(may_throw_exception=True),
        )
        def null_mask(x: InputArray) -> None:
            x.null_mask()

        msg = "Invalid to retrieve the null mask of a non-nullable array"
        arr = get_legate_runtime().create_array(
            ty.int32, (3, 2, 1), nullable=False
        )
        arr.fill(0)
        with pytest.raises(ValueError, match=msg):
            null_mask(arr)

    def test_register_invalid_param_variant(self) -> None:
        arr = get_legate_runtime().create_array(ty.int32)
        msg = re.escape(
            f"Type hint '{type(arr)}' is invalid, because it is "
            "impossible to deduce intent from it. Must use either "
            "Input/Output/Reduction variant"
        )
        with pytest.raises(TypeError, match=msg):

            @task
            def _(_: LogicalArray) -> None:
                pass

    def test_register_non_type_param(self) -> None:
        val = None
        msg = re.escape(
            f"Unhandled type annotation: {val}, expected this to be a "
            f"type, got {type(val)} instead"
        )
        with pytest.raises(AssertionError, match=msg):

            @task
            def _(_: None) -> None:
                pass

    def test_register_nargs(self) -> None:
        msg = re.escape(
            "'/', '*', '*args', '**kwargs' not yet allowed in parameter list"
        )
        with pytest.raises(NotImplementedError, match=msg):

            @task
            def _(*_: tuple[InputStore, ...]) -> None:
                pass

    def test_register_arbitrary_union_type(self) -> None:
        msg = re.escape(
            "Arbitrary union types not yet supported. Union types may "
            "only be 'SomeType | None' (order doesn't matter), "
            "'Union[SomeType, None]' (order doesn't matter), or "
            "'Optional[SomeType]'."
        )
        with pytest.raises(NotImplementedError, match=msg):

            @task
            def _(_: InputStore | InputArray) -> None:
                pass

        with pytest.raises(NotImplementedError, match=msg):

            @task
            def _(_: InputStore | InputArray | None) -> None:
                pass

    def test_constraint_invalid_var_name(self) -> None:
        msg = 'constraint argument "foo" not in set of parameters: '
        with pytest.raises(ValueError, match=msg):

            @task(
                variants=tuple(VariantCode), constraints=(align("foo", "bar"),)
            )
            def fill_task(in_store: InputStore, out: OutputStore) -> None:
                pass

    def test_constraint_on_scalar(self) -> None:
        msg = re.escape(
            "Could not find val in task arguments: "
            "((0, ()), (1, ('out',)), (2, ()))"
        )
        with pytest.raises(ValueError, match=msg):
            # looking at the comment in constraint._deduce_arg this constraint
            # is expected to fail, but it's not, so maybe this should be a bug?
            @task(
                variants=tuple(VariantCode), constraints=(align("val", "out"),)
            )
            def fill_task(val: int, out: OutputStore) -> None:
                pass

    def test_different_variant_func(self) -> None:
        runtime = get_legate_runtime()
        task_id = runtime.core_library.get_new_task_id()
        tc = TaskConfig(task_id, options=VariantOptions(has_allocations=True))

        shape = (3, 5, 2)

        def bind(out: tasks.OutputStore) -> None:
            pytest.fail("should never reach here")

        def bind_cpu(out: OutputStore) -> None:
            buf = out.create_output_buffer(shape)
            tasks.asarray(buf).fill(int(TaskTarget.CPU))

        def bind_gpu(out: OutputStore) -> None:
            buf = out.create_output_buffer(shape)
            tasks.asarray(buf).fill(int(TaskTarget.GPU))

        def bind_omp(out: OutputStore) -> None:
            buf = out.create_output_buffer(shape)
            tasks.asarray(buf).fill(int(TaskTarget.OMP))

        out_store = runtime.create_store(ty.int32, ndim=len(shape))

        bind_task = PyTask(func=bind, variants=[], options=tc)
        bind_task.cpu_variant(bind_cpu)
        bind_task.gpu_variant(bind_gpu)
        bind_task.omp_variant(bind_omp)
        exp_arr: NDArray[Any] = np.ndarray(shape, dtype=np.int32)
        exp_arr.fill(int(runtime.machine.preferred_target))
        bind_task(out_store)
        runtime.issue_execution_fence(block=True)
        np.testing.assert_allclose(
            np.asarray(out_store.get_physical_store()), exp_arr
        )

    def test_update_variants_on_completed_task(self) -> None:
        def foo() -> None:
            pass

        task = PyTask(func=foo, variants=[VariantCode.CPU])
        task.complete_registration()
        msg = re.escape(
            f"Task (id: {task.task_id}) has already completed "
            "registration and cannot update its variants"
        )
        with pytest.raises(RuntimeError, match=msg):
            task.gpu_variant(print)

    def test_empty_variants(self) -> None:
        def foo() -> None:
            pass

        msg = "Task has no registered variants"
        task = PyTask(func=foo, variants=[])
        with pytest.raises(ValueError, match=msg):
            _ = task.complete_registration()

    @pytest.mark.skipif(
        TaskTarget.OMP in get_legate_runtime().machine.valid_targets,
        reason="CPU/GPU only test",
    )
    def test_unregistered_variant(self) -> None:
        def foo() -> None:
            pass

        pytask = PyTask(func=foo, variants=(VariantCode.OMP,))
        msg = (
            "does not have any valid variant for the current machine "
            "configuration"
        )
        with pytest.raises(ValueError, match=msg):
            pytask()

    def test_store_get_partition(self) -> None:
        runtime = get_legate_runtime()
        shape = (10000 * 32,)
        in_arr_np = np.ones(shape=shape, dtype=np.int32)
        in_store = runtime.create_store_from_buffer(
            ty.int32, in_arr_np.shape, in_arr_np, False
        )
        out_store = runtime.create_store(ty.int32, shape)
        auto_task = runtime.create_auto_task(
            tasks.copy_store_task.library, tasks.copy_store_task.task_id
        )

        auto_task.add_input(in_store)
        auto_task.add_output(out_store)
        auto_task.execute()
        runtime.issue_execution_fence(block=True)

        sum_input1_np = np.full(shape, 2, dtype=np.int32)
        sum_input1_store = runtime.create_store_from_buffer(
            ty.int32, sum_input1_np.shape, sum_input1_np, False
        )

        color_shape: tuple[int, ...] = (2, 1)
        if out_store.partition is not None:
            partition = sum_input1_store.partition_by_tiling(
                shape, out_store.partition.color_shape
            )

            assert partition is not None
            assert partition.color_shape == out_store.partition.color_shape
            color_shape = partition.color_shape

        sum_output_store = runtime.create_store(ty.int32, shape)

        manual_task = runtime.create_manual_task(
            tasks.sum_two_inputs_task.library,
            tasks.sum_two_inputs_task.task_id,
            color_shape,
        )

        manual_task.add_input(sum_input1_store)
        manual_task.add_input(out_store)
        manual_task.add_output(sum_output_store)
        manual_task.execute()
        runtime.issue_execution_fence(block=True)

        result_arr = np.asarray(sum_output_store.get_physical_store())
        expected_arr = np.full(shape, 3, dtype=np.int32)
        np.testing.assert_allclose(result_arr, expected_arr)

    @pytest.mark.parametrize(
        "mode", [StreamingMode.OFF, StreamingMode.RELAXED], ids=repr
    )
    @pytest.mark.parametrize("factor", [1, 2, 43])
    def test_parallel_tasks_with_scalar(
        self, mode: StreamingMode, factor: int
    ) -> None:
        shape = (100, 10, 100)
        arr1, store1 = utils.create_np_array_and_store(
            ty.int32, shape, func=np.zeros
        )
        arr2, store2 = utils.create_np_array_and_store(
            ty.int32, shape, func=np.zeros
        )

        class Counter:
            def __init__(self, start: int) -> None:
                self.idx = start

            def increment(self) -> int:
                self.idx += 1
                return self.idx

        counters = {1: Counter(0), 2: Counter(1024)}

        @task(variants=tuple(VariantCode))
        def fill_task(out: OutputStore, counter: int) -> None:
            out_arr = tasks.asarray(out)
            out_arr[:] = counters[counter].increment()

        with Scope(
            parallel_policy=ParallelPolicy(
                streaming_mode=mode, overdecompose_factor=factor
            )
        ):
            fill_task(store1, 1)
            fill_task(store2, 2)

        get_legate_runtime().issue_execution_fence(block=True)
        assert (arr1 <= counters[1].idx).all()
        assert (arr2 <= counters[2].idx).all()

    @pytest.mark.parametrize(
        "mode", [StreamingMode.OFF, StreamingMode.RELAXED], ids=repr
    )
    @pytest.mark.parametrize("factor", [1, 2, 43])
    @pytest.mark.skipif(is_multi_node(), reason="single node only")
    def test_parallel_tasks(self, mode: StreamingMode, factor: int) -> None:
        runtime = get_legate_runtime()
        subregions = len(runtime.machine) * factor
        shape = (2 * subregions,)
        arr1 = np.arange(2 * subregions)
        store1 = runtime.create_store_from_buffer(
            ty.int64, shape, arr1, read_only=False
        )
        out1 = runtime.create_store(ty.int64, shape)
        out2 = runtime.create_store(ty.int64, shape)

        with Scope(
            parallel_policy=ParallelPolicy(
                overdecompose_factor=factor, streaming_mode=mode
            )
        ):
            assert Scope.parallel_policy().streaming_mode == mode
            assert Scope.parallel_policy().overdecompose_factor == factor
            tasks.copy_store_task(store1, out1)
            tasks.zeros_task(out2)

        get_legate_runtime().issue_execution_fence(block=True)
        np.testing.assert_allclose(np.asarray(out1), arr1)
        assert np.all(np.asarray(out2) == 0)

    def test_flush_scheduling_within_stream(self) -> None:
        msg = "flush_scheduling_window called from inside a streaming scope"
        with (
            Scope(
                parallel_policy=ParallelPolicy(
                    streaming_mode=StreamingMode.STRICT, overdecompose_factor=2
                )
            ),
            pytest.raises(ValueError, match=msg),
        ):
            get_legate_runtime().issue_execution_fence(block=True)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
