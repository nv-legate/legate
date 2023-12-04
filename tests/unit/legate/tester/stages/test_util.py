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

"""Consolidate test configuration from command-line and environment.

"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from legate.tester.config import Config
from legate.tester.logger import LOG
from legate.tester.stages import util as m
from legate.tester.test_system import ProcessResult
from legate.util.ui import failed, passed, shell, skipped, timeout


def test_StageResult() -> None:
    procs = [ProcessResult(f"run{i}", Path(f"test{i}")) for i in range(10)]
    procs[2].returncode = 10
    procs[7].returncode = -2

    result = m.StageResult(procs=procs, time=timedelta(0))

    assert result.total == 10
    assert result.passed == 8


class Test_adjust_workers:
    @pytest.mark.parametrize("n", (1, 5, 100))
    def test_None_requested(self, n: int) -> None:
        assert m.adjust_workers(n, None) == n

    @pytest.mark.parametrize("n", (1, 2, 9))
    def test_requested(self, n: int) -> None:
        assert m.adjust_workers(10, n) == n

    def test_negative_requested(self) -> None:
        with pytest.raises(ValueError):
            assert m.adjust_workers(10, -1)

    def test_zero_requested(self) -> None:
        with pytest.raises(RuntimeError):
            assert m.adjust_workers(10, 0)

    def test_zero_computed(self) -> None:
        with pytest.raises(RuntimeError):
            assert m.adjust_workers(0, None)

    def test_requested_too_large(self) -> None:
        with pytest.raises(RuntimeError):
            assert m.adjust_workers(10, 11)


class Test_log_proc:
    @pytest.mark.parametrize("returncode", (-23, -1, 0, 1, 17))
    def test_skipped(self, returncode: int) -> None:
        config = Config([])
        proc = ProcessResult(
            "proc", Path("proc"), skipped=True, returncode=returncode
        )

        LOG.clear()
        m.log_proc("foo", proc, config, verbose=False)

        assert LOG.lines == (skipped(f"(foo) {proc.test_file}"),)

    def test_passed(self) -> None:
        config = Config([])
        proc = ProcessResult("proc", Path("proc"))

        LOG.clear()
        m.log_proc("foo", proc, config, verbose=False)

        assert LOG.lines == (passed(f"(foo) {proc.test_file}"),)

    def test_passed_verbose(self) -> None:
        config = Config([])
        proc = ProcessResult("proc", Path("proc"), output="foo\nbar")
        details = proc.output.split("\n")

        LOG.clear()
        m.log_proc("foo", proc, config, verbose=True)

        assert LOG.lines == tuple(
            passed(f"(foo) {proc.test_file}", details=details).split("\n")
        )

    @pytest.mark.parametrize("returncode", (-23, -1, 1, 17))
    def test_failed(self, returncode: int) -> None:
        config = Config([])
        proc = ProcessResult("proc", Path("proc"), returncode=returncode)

        LOG.clear()
        m.log_proc("foo", proc, config, verbose=False)

        assert LOG.lines == (
            failed(f"(foo) {proc.test_file}", exit_code=returncode),
        )

    @pytest.mark.parametrize("returncode", (-23, -1, 1, 17))
    def test_failed_verbose(self, returncode: int) -> None:
        config = Config([])
        proc = ProcessResult(
            "proc", Path("proc"), returncode=returncode, output="foo\nbar"
        )
        details = proc.output.split("\n")

        LOG.clear()
        m.log_proc("foo", proc, config, verbose=True)

        assert LOG.lines == tuple(
            failed(
                f"(foo) {proc.test_file}",
                details=details,
                exit_code=returncode,
            ).split("\n")
        )

    def test_timeout(self) -> None:
        config = Config([])
        proc = ProcessResult("proc", Path("proc"), timeout=True)

        LOG.clear()
        m.log_proc("foo", proc, config, verbose=False)

        assert LOG.lines == (timeout(f"(foo) {proc.test_file}"),)

    def test_dry_run(self) -> None:
        config = Config(["test.py", "--dry-run"])
        proc = ProcessResult("proc", Path("proc"))

        LOG.clear()
        m.log_proc("foo", proc, config, verbose=False)

        assert LOG.lines == (
            shell(proc.invocation),
            passed(f"(foo) {proc.test_file}"),
        )

    def test_debug(self) -> None:
        config = Config(["test.py", "--debug"])
        proc = ProcessResult("proc", Path("proc"))

        LOG.clear()
        m.log_proc("foo", proc, config, verbose=False)

        assert LOG.lines == (
            shell(proc.invocation),
            passed(f"(foo) {proc.test_file}"),
        )

    def test_time(self) -> None:
        config = Config(["test.py", "--debug"])
        start = datetime.now()
        end = start + timedelta(seconds=2.41)

        proc = ProcessResult("proc", Path("proc"), start=start, end=end)

        LOG.clear()
        m.log_proc("foo", proc, config, verbose=False)

        assert LOG.lines[0] == shell(proc.invocation)
        assert LOG.lines[1].startswith(passed("(foo) 2.41s {"))
        assert LOG.lines[1].endswith(f"}} {proc.test_file}")
