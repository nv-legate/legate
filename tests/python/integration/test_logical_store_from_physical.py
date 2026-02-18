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
    runtime = get_legate_runtime()

    logical = runtime.create_logical_store_from_physical(inp)

    result = np.asarray(result_out)
    result[0] = logical.ndim
    result[1] = logical.extents[0]
    result[2] = 1 if logical.type == ty.int32 else 0


@task
def verify_api_2d_task(inp: InputStore, result_out: OutputStore) -> None:
    runtime = get_legate_runtime()

    logical = runtime.create_logical_store_from_physical(inp)

    result = np.asarray(result_out)
    result[0] = logical.ndim
    result[1] = logical.extents[0]
    result[2] = logical.extents[1]


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
