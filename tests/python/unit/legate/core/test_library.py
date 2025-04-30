# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from legate.core import get_legate_runtime


class TestLibrary:
    def test_name(self) -> None:
        lib_name = "test_library.test_name.foo"
        lib, _ = get_legate_runtime().find_or_create_library(lib_name)
        assert lib.name == lib_name
