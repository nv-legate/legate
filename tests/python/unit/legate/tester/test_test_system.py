# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Consolidate test configuration from command-line and environment."""

from __future__ import annotations

from datetime import timedelta
from subprocess import PIPE, STDOUT
from typing import TYPE_CHECKING
from unittest.mock import ANY

import psutil

import pytest

from legate.tester import test_system as m

if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from pytest_mock import MockerFixture


class TestProcessResult:
    def test_default(self) -> None:
        ret = m.ProcessResult("proc", "proc_display")
        assert ret.invocation == "proc"
        assert ret.test_display == "proc_display"
        assert ret.time is None
        assert not ret.skipped
        assert not ret.timeout
        assert ret.returncode == 0
        assert ret.output == ""
        assert ret.passed

    def test_passed_skipped(self) -> None:
        ret = m.ProcessResult("proc", "proc_display", skipped=True)
        assert ret.passed

    def test_passed_return_zero(self) -> None:
        ret = m.ProcessResult("proc", "proc_display", returncode=0)
        assert ret.passed

    def test_passed_return_nonzero(self) -> None:
        ret = m.ProcessResult("proc", "proc_display", returncode=1)
        assert not ret.passed

    def test_passed_timeout(self) -> None:
        ret = m.ProcessResult("proc", "proc_display", timeout=True)
        assert not ret.passed


@pytest.fixture
def mock_popen(mocker: MockerFixture) -> MagicMock:
    mock_popen = mocker.patch.object(m, "Popen")

    # Configure the mock to return a tuple from communicate()
    # and configure the context manager return value
    mock_proc = mock_popen.return_value.__enter__.return_value
    mock_proc.communicate.return_value = ("", None)
    mock_proc.returncode = 0
    return mock_popen


CMD = "legate script.py --cpus 4"


class TestSystem:
    def test_init(self) -> None:
        s = m.TestSystem()
        assert s.dry_run is False

    def test_run(self, mock_popen: MagicMock) -> None:
        s = m.TestSystem()

        result = s.run(CMD.split(), "test/file")
        mock_popen.assert_called_once_with(
            CMD.split(),
            cwd=None,
            env=ANY,
            stdout=PIPE,
            stderr=STDOUT,
            errors="replace",
        )

        assert result.invocation == CMD
        assert result.test_display == "test/file"
        assert result.time is not None
        assert result.time > timedelta(0)
        assert not result.skipped
        assert not result.timeout

    def test_run_with_returncode(self) -> None:
        s = m.TestSystem()

        false_cmd = "false"
        result = s.run(false_cmd.split(), "test/file")

        assert result.invocation == false_cmd
        assert result.test_display == "test/file"
        assert result.time is not None
        assert result.time > timedelta(0)
        assert not result.skipped
        assert not result.timeout
        assert result.returncode != 0
        assert result.output is not None
        assert not result.passed

    def test_run_with_stdout(self) -> None:
        s = m.TestSystem()

        echo_cmd = "echo Hello"
        result = s.run(echo_cmd.split(), "test/file")

        assert result.invocation == echo_cmd
        assert result.test_display == "test/file"
        assert result.time is not None
        assert result.time > timedelta(0)
        assert not result.skipped
        assert not result.timeout
        assert result.returncode == 0
        assert result.output == "Hello\n"
        assert result.passed

    def test_dry_run(self, mock_popen: MagicMock) -> None:
        s = m.TestSystem(dry_run=True)

        result = s.run(CMD.split(), "test/file")
        mock_popen.assert_not_called()

        assert result.output == ""
        assert result.skipped

    def test_timeout(self) -> None:
        s = m.TestSystem()

        result = s.run(["sleep", "2"], "test/file", timeout=1)

        assert result.timeout
        assert not result.skipped
        assert not result.passed

    def test_memory(self) -> None:
        s = m.TestSystem()

        assert s.memory == psutil.virtual_memory().total
