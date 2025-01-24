# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Consolidate test configuration from command-line and environment."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from legate.tester.config import Config
from legate.tester.logger import LOG
from legate.tester.project import Project
from legate.tester.stages import util as m
from legate.tester.test_system import ProcessResult
from legate.util.ui import failed, passed, shell, skipped, timeout

PROJECT = Project()


def test_StageResult() -> None:
    procs = [ProcessResult(f"run{i}", f"test{i}") for i in range(10)]
    procs[2].returncode = 10
    procs[7].returncode = -2
    procs[8].timeout = True

    result = m.StageResult(procs=procs, time=timedelta(0))

    assert result.total == 10
    assert result.passed == 7


class Test_adjust_workers:
    @pytest.mark.parametrize("n", (1, 5, 100))
    def test_None_requested(self, n: int) -> None:
        assert m.adjust_workers(n, None) == n

    @pytest.mark.parametrize("n", (1, 2, 9))
    def test_requested(self, n: int) -> None:
        assert m.adjust_workers(10, n) == n

    def test_negative_requested(self) -> None:
        with pytest.raises(
            ValueError, match="requested workers must be non-negative"
        ):
            assert m.adjust_workers(10, -1)

    def test_zero_requested(self) -> None:
        with pytest.raises(
            RuntimeError, match="requested workers must not be zero"
        ):
            assert m.adjust_workers(10, 0)

    def test_zero_adjusted(self) -> None:
        with pytest.raises(
            RuntimeError, match="Current configuration results in zero workers"
        ):
            assert m.adjust_workers(0, None)

    def test_zero_adjusted_with_detail(self) -> None:
        with pytest.raises(
            RuntimeError,
            match="Current configuration results in zero workers "
            r"\[details: foo bar\]",
        ):
            assert m.adjust_workers(0, None, detail="foo bar")

    def test_requested_too_large(self) -> None:
        with pytest.raises(
            RuntimeError,
            match=r"Requested workers \(11\) is greater than "
            r"computed workers \(10\)",
        ):
            assert m.adjust_workers(10, 11)


class Test_log_proc:
    @pytest.mark.parametrize("returncode", (-23, -1, 0, 1, 17))
    def test_skipped(self, returncode: int) -> None:
        config = Config([], project=PROJECT)
        proc = ProcessResult(
            "proc", "proc_display", skipped=True, returncode=returncode
        )

        LOG.clear()
        m.log_proc("foo", proc, config, verbose=False)

        assert LOG.lines == (skipped(f"(foo) {proc.test_display}").plain,)

    def test_passed(self) -> None:
        config = Config([], project=PROJECT)
        proc = ProcessResult("proc", "proc_display")

        LOG.clear()
        m.log_proc("foo", proc, config, verbose=False)

        assert LOG.lines == (passed(f"(foo) {proc.test_display}").plain,)

    def test_passed_verbose(self) -> None:
        config = Config([], project=PROJECT)
        proc = ProcessResult("proc", "proc_display", output="foo\nbar")
        details = proc.output.split("\n")

        LOG.clear()
        m.log_proc("foo", proc, config, verbose=True)

        assert LOG.lines == tuple(
            passed(f"(foo) {proc.test_display}", details=details).plain.split(
                "\n"
            )[:-1]
        )

    @pytest.mark.parametrize("returncode", (-23, -1, 1, 17))
    def test_failed(self, returncode: int) -> None:
        config = Config([], project=PROJECT)
        proc = ProcessResult("proc", "proc_display", returncode=returncode)

        LOG.clear()
        m.log_proc("foo", proc, config, verbose=False)

        assert LOG.lines == (
            failed(f"(foo) {proc.test_display}", exit_code=returncode).plain,
        )

    @pytest.mark.parametrize("returncode", (-23, -1, 1, 17))
    def test_failed_verbose(self, returncode: int) -> None:
        config = Config([], project=PROJECT)
        proc = ProcessResult(
            "proc", "proc_display", returncode=returncode, output="foo\nbar"
        )
        details = proc.output.split("\n")

        LOG.clear()
        m.log_proc("foo", proc, config, verbose=True)

        assert LOG.lines == tuple(
            failed(
                f"(foo) {proc.test_display}",
                details=details,
                exit_code=returncode,
            ).plain.split("\n")[:-1]
        )

    def test_timeout(self) -> None:
        config = Config([], project=PROJECT)
        proc = ProcessResult("proc", "proc_display", timeout=True)

        LOG.clear()
        m.log_proc("foo", proc, config, verbose=False)

        assert LOG.lines == (timeout(f"(foo) {proc.test_display}").plain,)

    def test_timeout_verbose(self) -> None:
        config = Config([], project=PROJECT)
        start = datetime.now()
        end = start + timedelta(seconds=45)
        proc = ProcessResult(
            "proc",
            "proc_display",
            start=start,
            end=end,
            timeout=True,
            output="foo\nbar",
        )
        duration = m.format_duration(start=start, end=end)
        details = proc.output.split("\n")

        LOG.clear()
        m.log_proc("foo", proc, config, verbose=True)

        assert LOG.lines == tuple(
            timeout(
                f"(foo){duration} {proc.test_display}", details=details
            ).plain.split("\n")[:-1]
        )

    def test_dry_run(self) -> None:
        config = Config(["test.py", "--dry-run"], project=PROJECT)
        proc = ProcessResult("proc", "proc_display")

        LOG.clear()
        m.log_proc("foo", proc, config, verbose=False)

        assert LOG.lines == (
            "",
            shell(proc.invocation).plain,
            passed(f"(foo) {proc.test_display}").plain,
        )

    def test_debug(self) -> None:
        config = Config(["test.py", "--debug"], project=PROJECT)
        proc = ProcessResult("proc", "proc_display")

        LOG.clear()
        m.log_proc("foo", proc, config, verbose=False)

        assert LOG.lines == (
            "",
            shell(proc.invocation).plain,
            passed(f"(foo) {proc.test_display}").plain,
        )

    def test_time(self) -> None:
        config = Config(["test.py", "--debug"], project=PROJECT)
        start = datetime.now()
        end = start + timedelta(seconds=2.41)

        proc = ProcessResult("proc", "proc_display", start=start, end=end)

        LOG.clear()
        m.log_proc("foo", proc, config, verbose=False)

        duration = m.format_duration(start=start, end=end)
        assert LOG.lines == (
            "",
            shell(proc.invocation).plain,
            passed(f"(foo){duration} {proc.test_display}").plain,
        )
