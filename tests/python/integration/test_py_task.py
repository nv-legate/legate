# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

import pytest

from legate.core import (
    ImageComputationHint,
    LogicalArray,
    Scalar,
    TaskTarget,
    Type,
    VariantCode,
    VariantOptions,
    align,
    bloat,
    broadcast,
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
    ReductionStore,
    task,
)

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

    def test_prepare_nullable_param(self) -> None:
        runtime = get_legate_runtime()
        arr = runtime.create_array(ty.int32, (1, 2, 3), nullable=True)
        msg = (
            "Legate data interface objects with nullable stores are "
            "unsupported"
        )
        with pytest.raises(NotImplementedError, match=msg):
            tasks.fill_task.prepare_call(arr, Scalar(None))

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


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
