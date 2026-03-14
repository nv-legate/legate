# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import pytest

from legate.core import get_legate_runtime, types as ty
from legate.core.task import InputStore, OutputStore, task

from .utils.utils import is_multi_node


@task
def verify_api_task(inp: InputStore, result_out: OutputStore) -> None:
    logical = inp.to_logical_store()

    result = np.asarray(result_out)
    result[0] = logical.ndim
    result[1] = logical.extents[0]
    result[2] = 1 if logical.type == ty.int32 else 0


@task
def verify_api_2d_task(inp: InputStore, result_out: OutputStore) -> None:
    logical = inp.to_logical_store()

    result = np.asarray(result_out)
    result[0] = logical.ndim
    result[1] = logical.extents[0]
    result[2] = logical.extents[1]


@task
def inner_double_task(inp: InputStore, out: OutputStore) -> None:
    in_arr = np.asarray(inp)
    out_arr = np.asarray(out)
    out_arr[:] = in_arr * 2


@task
def outer_nested_task(inp: InputStore, out: OutputStore) -> None:
    logical_in = inp.to_logical_store()
    logical_out = out.to_logical_store()

    inner_double_task(logical_in, logical_out)


@task
def inner_add_task(inp: InputStore, out: OutputStore, value: int) -> None:
    in_arr = np.asarray(inp)
    out_arr = np.asarray(out)
    out_arr[:] = in_arr + value


@task
def outer_chained_task(
    inp: InputStore, out: OutputStore, intermediate: OutputStore
) -> None:
    logical_in = inp.to_logical_store()
    logical_out = out.to_logical_store()
    logical_intermediate = intermediate.to_logical_store()

    inner_double_task(logical_in, logical_intermediate)
    inner_add_task(logical_intermediate, logical_out, 10)


@pytest.mark.skipif(is_multi_node(), reason="Store partitioned across ranks")
class TestNestedExecution:
    def test_create_logical_from_physical(self) -> None:
        runtime = get_legate_runtime()

        store = runtime.create_store(ty.int32, shape=(10,))
        arr = np.asarray(store)
        arr[:] = range(10)

        result_store = runtime.create_store(ty.int32, shape=(3,))

        verify_api_task(store, result_store)

        result = np.asarray(result_store)
        assert result[0] == 1
        assert result[1] == 10
        assert result[2] == 1

    def test_create_logical_from_physical_2d(self) -> None:
        runtime = get_legate_runtime()

        store = runtime.create_store(ty.float32, shape=(3, 4))
        arr = np.asarray(store)
        arr[:] = np.arange(12, dtype=np.float32).reshape(3, 4)

        result_store = runtime.create_store(ty.int32, shape=(3,))

        verify_api_2d_task(store, result_store)

        result = np.asarray(result_store)
        assert result[0] == 2
        assert result[1] == 3
        assert result[2] == 4

    def test_nested_task_simple(self) -> None:
        """Test that one @task can call another @task as a nested task."""
        runtime = get_legate_runtime()

        in_store = runtime.create_store(ty.int32, shape=(10,))
        in_arr = np.asarray(in_store)
        in_arr[:] = np.arange(10, dtype=np.int32)

        out_store = runtime.create_store(ty.int32, shape=(10,))

        outer_nested_task(in_store, out_store)

        result = np.asarray(out_store)
        expected = np.arange(10, dtype=np.int32) * 2
        np.testing.assert_array_equal(result, expected)

    def test_nested_task_chained(self) -> None:
        """Test chaining multiple nested tasks within an outer task."""
        runtime = get_legate_runtime()

        in_store = runtime.create_store(ty.int32, shape=(5,))
        in_arr = np.asarray(in_store)
        in_arr[:] = np.array([1, 2, 3, 4, 5], dtype=np.int32)

        out_store = runtime.create_store(ty.int32, shape=(5,))
        intermediate_store = runtime.create_store(ty.int32, shape=(5,))

        outer_chained_task(in_store, out_store, intermediate_store)

        result = np.asarray(out_store)
        expected = np.array([1, 2, 3, 4, 5], dtype=np.int32) * 2 + 10
        np.testing.assert_array_equal(result, expected)
