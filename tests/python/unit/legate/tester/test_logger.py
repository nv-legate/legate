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

"""Consolidate test configuration from command-line and environment.

"""
from __future__ import annotations

from legate.tester import logger as m

TEST_LINES = (
    "line 1",
    "[red]foo[/]",
    "bar",
    "\033[1msome ANSI\033[0m",
    "last line",
)

SCRUBBED_TEST_LINES = (
    "line 1",
    "foo",
    "bar",
    "some ANSI",
    "last line",
)


class TestLogger:
    def test_init(self) -> None:
        log = m.Log()
        assert log.lines == ()
        assert log.dump() == "\n"

    def test_record_lines(self) -> None:
        log = m.Log()
        log.record(*TEST_LINES)
        assert log.lines == TEST_LINES
        assert log.dump() == "\n".join(SCRUBBED_TEST_LINES)

    def test_record_line_with_newlines(self) -> None:
        log = m.Log()
        log.record("\n".join(TEST_LINES))
        assert log.lines == TEST_LINES
        assert log.dump() == "\n".join(SCRUBBED_TEST_LINES)

    def test_call(self) -> None:
        log = m.Log()
        log(*TEST_LINES)
        assert log.lines == TEST_LINES
        assert log.dump() == "\n".join(SCRUBBED_TEST_LINES)

    def test_dump_index(self) -> None:
        log = m.Log()
        log.record(*TEST_LINES)
        assert log.dump(start=1, end=3) == "\n".join(SCRUBBED_TEST_LINES[1:3])

    def test_clear(self) -> None:
        log = m.Log()
        log.record(*TEST_LINES)
        assert len(log.lines) > 0
        log.clear()
        assert len(log.lines) == 0


def test_LOG() -> None:
    assert isinstance(m.LOG, m.Log)
