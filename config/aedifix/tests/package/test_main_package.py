# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest

from ...package.main_package import DebugConfigureValue

if TYPE_CHECKING:
    from collections.abc import Iterator

ALL_DEBUG_CONFIGURE_FLAGS = (
    DebugConfigureValue.NONE,
    DebugConfigureValue.DEBUG_FIND,
    DebugConfigureValue.TRACE,
    DebugConfigureValue.TRACE_EXPAND,
)

ALL_DEBUG_CMAKE_FLAGS = ("", "--debug-find", "--trace", "--trace-expand")


def gen_expected_flags() -> Iterator[list[str]]:
    for i in range(len(ALL_DEBUG_CMAKE_FLAGS)):
        ret = list(ALL_DEBUG_CMAKE_FLAGS[: i + 1])
        ret.remove("")
        yield ret


class TestDebugConfigureValue:
    @pytest.mark.parametrize(
        ("val", "expected"),
        list(zip(ALL_DEBUG_CONFIGURE_FLAGS, ALL_DEBUG_CMAKE_FLAGS)),
    )
    def test_flag_matches(
        self, val: DebugConfigureValue, expected: str
    ) -> None:
        assert val.to_flag() == expected

    def test_help_str(self) -> None:
        help_str = DebugConfigureValue.help_str()
        for cmake_flg in ALL_DEBUG_CMAKE_FLAGS:
            assert cmake_flg in help_str
        for flg in ALL_DEBUG_CONFIGURE_FLAGS:
            assert flg.to_flag() in help_str
            assert str(flg) in help_str

    @pytest.mark.parametrize(
        ("val", "expected"),
        list(zip(ALL_DEBUG_CONFIGURE_FLAGS, gen_expected_flags())),
    )
    def test_to_flags(
        self, val: DebugConfigureValue, expected: list[str]
    ) -> None:
        assert val.to_flags() == expected


if __name__ == "__main__":
    sys.exit(pytest.main())
