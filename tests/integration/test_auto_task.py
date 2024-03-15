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

from typing import Any, Type

import numpy as np
import pytest
from utils import tasks
from utils.data import ARRAY_TYPES, SCALAR_VALS

from legate.core import LogicalArray, Scalar, get_legate_runtime, types as ty
from legate.core.task import task


class TestAutoTask:

    def test_create_auto_task(self) -> None:
        runtime = get_legate_runtime()
        library = runtime.core_library
        runtime.create_auto_task(library, tasks.basic_task.task_id)

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
    def test_input_output(self, shape: tuple[int]) -> None:
        runtime = get_legate_runtime()
        auto_task = runtime.create_auto_task(
            runtime.core_library, tasks.copy_store_task.task_id
        )

        in_arr_np = np.ndarray(shape=shape, dtype=np.int32)
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
        shape = (3, 1, 3)
        dtype_np = dtype.to_numpy_dtype()
        runtime = get_legate_runtime()
        auto_task = runtime.create_auto_task(
            runtime.core_library, tasks.fill_task.task_id
        )
        out_store = runtime.create_store(dtype, shape)
        auto_task.add_output(out_store)

        arr_np = np.full(shape, val, dtype=dtype_np)

        auto_task.add_scalar_arg(Scalar(val, dtype))
        auto_task.execute()
        runtime.issue_execution_fence(block=True)
        np.testing.assert_allclose(
            arr_np,
            np.asarray(out_store.get_physical_store().get_inline_allocation()),
        )

    @pytest.mark.parametrize("size", [9, 101, 32768], ids=str)
    def test_tuple_scalar_arg(self, size: int) -> None:
        shape = (size,)
        runtime = get_legate_runtime()
        auto_task = runtime.create_auto_task(
            runtime.core_library, tasks.copy_np_array_task.task_id
        )
        out_store = runtime.create_store(ty.int32, shape)
        auto_task.add_output(out_store)

        arr_np = np.ndarray(shape=shape, dtype=np.int32)
        auto_task.add_broadcast(out_store)

        auto_task.add_scalar_arg(arr_np, (ty.int32,))
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
        arg2_array = LogicalArray.from_store(arg1_store)
        out_store = runtime.create_store(ty.float64, (5, 4, 3, 2))
        auto_task.add_input(arg2_array)
        auto_task.add_input(arg2_store)
        auto_task.add_output(out_store)
        auto_task.execute()
        runtime.issue_execution_fence(block=True)
        np.testing.assert_allclose(
            arg1_np + arg2_np + out_np,
            np.asarray(out_store.get_physical_store().get_inline_allocation()),
        )

    @pytest.mark.parametrize(
        "shape",
        [(1, 2, 1), (2, 101, 10), (3, 4096, 12), (65535, 1, 2)],
        ids=str,
    )
    @pytest.mark.parametrize(
        "accessed",
        [
            True,
            pytest.param(
                False,
                marks=pytest.mark.xfail(
                    reason="crashes application", run=False
                ),
            ),
        ],
        ids=["accessed", "unaccessed"],
    )
    def test_uninitialized_input_store(self, shape, accessed) -> None:
        runtime = get_legate_runtime()
        auto_task = runtime.create_auto_task(
            runtime.core_library, tasks.copy_store_task.task_id
        )

        in_store = runtime.create_store(ty.int32, shape=shape)

        if accessed:
            in_store.get_physical_store().get_inline_allocation()

        out_store = runtime.create_store(ty.int32, shape)

        auto_task.add_input(in_store)
        auto_task.add_output(out_store)
        auto_task.add_alignment(out_store, in_store)
        # issue 465: crashes application if input store is not accessed prior
        auto_task.execute()
        runtime.issue_execution_fence(block=True)
        np.testing.assert_allclose(
            in_store.get_physical_store().get_inline_allocation(),
            out_store.get_physical_store().get_inline_allocation(),
        )


class TestAutoTaskErrors:
    def test_nonexistent_task_id(self) -> None:
        runtime = get_legate_runtime()
        msg = "does not have task"
        with pytest.raises(IndexError, match=msg):
            runtime.create_auto_task(runtime.core_library, 111)

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

    def test_add_scalar_arg_without_type(self) -> None:
        runtime = get_legate_runtime()
        auto_task = runtime.create_auto_task(
            runtime.core_library, tasks.basic_task.task_id
        )
        msg = "Data type must be given if the value is not a Scalar object"
        with pytest.raises(ValueError, match=msg):
            auto_task.add_scalar_arg(123)

    @pytest.mark.parametrize(
        "dtype", [(ty.int32, ty.int64), (np.int32,)], ids=str
    )
    def test_unsupported_scalar_arg_type(self, dtype: tuple[Any]) -> None:
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
        with pytest.raises(IndexError, match="_Map_base::at"):
            auto_task.execute()
        runtime.issue_execution_fence(block=True)

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
        # issue 384/440
        # reusing auto task should not be allowed
        # actual behavior to be updated after reuse check is implemented
        try:
            auto_task.execute()
        except Exception as exc:
            assert msg not in str(exc)
        runtime.issue_execution_fence(block=True)
