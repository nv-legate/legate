# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from legate.core import ResourceConfig, get_legate_runtime


class TestLibrary:
    def test_properties(self) -> None:
        lib_name = "test_library.test_name.foo"
        lib, _ = get_legate_runtime().find_or_create_library(lib_name)
        assert lib.name == lib_name
        # just touching for code coverage
        _ = lib.raw_handle

    def test_get_task_id(self) -> None:
        runtime = get_legate_runtime()
        lib, _ = runtime.find_or_create_library(
            "test_get_task_ids", config=ResourceConfig(max_dyn_tasks=1)
        )
        local_id = lib.get_new_task_id()
        assert isinstance(lib.get_task_id(local_id), int)

    def test_get_reduction_op_id(self) -> None:
        lib = get_legate_runtime().create_library("test_get_reduction_op_id")
        assert isinstance(lib.get_reduction_op_id(-1), int)


class TestLibraryErrors:
    def test_task_id_overflow(self) -> None:
        runtime = get_legate_runtime()
        test_lib = runtime.create_library("test_task_id_overflow")
        with pytest.raises(OverflowError, match="The scope ran out of IDs"):
            test_lib.get_new_task_id()

    def test_reduction_op_id_overflow(self) -> None:
        lib = get_legate_runtime().create_library(
            "test_get_reduction_op_id_overflow"
        )
        msg = "Maximum local ID is -1 but received a local ID 0"
        with pytest.raises(IndexError, match=msg):
            lib.get_reduction_op_id(0)
