# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

import legate.util.fs as m

HEADER_PATH = Path(__file__).parent / "sample_header.h"


def test_read_c_define_hit() -> None:
    assert m.read_c_define(HEADER_PATH, "FOO") == "10"
    assert m.read_c_define(HEADER_PATH, "BAR") == '"bar"'


def test_read_c_define_miss() -> None:
    assert m.read_c_define(HEADER_PATH, "JUNK") is None


CMAKE_CACHE_PATH = Path(__file__).parent / "sample_cmake_cache.txt"


def test_read_cmake_cache_value_hit() -> None:
    assert (
        m.read_cmake_cache_value(CMAKE_CACHE_PATH, "Legion_SOURCE_DIR:STATIC=")
        == '"foo/bar"'
    )
    assert (
        m.read_cmake_cache_value(
            CMAKE_CACHE_PATH, "FIND_LEGATE_CORE_CPP:BOOL=OFF"
        )
        == "OFF"
    )


def test_read_cmake_cache_value_miss() -> None:
    with pytest.raises(RuntimeError):
        assert m.read_cmake_cache_value(CMAKE_CACHE_PATH, "JUNK") is None
