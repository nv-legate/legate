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

"""Consolidate test configuration from command-line and environment."""

from __future__ import annotations

from datetime import timedelta
from subprocess import CompletedProcess
from typing import TYPE_CHECKING

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
def mock_subprocess_run(mocker: MockerFixture) -> MagicMock:
    return mocker.patch.object(m, "stdlib_run")


CMD = "legate script.py --cpus 4"


class TestSystem:
    def test_init(self) -> None:
        s = m.TestSystem()
        assert s.dry_run is False

    def test_run(self, mock_subprocess_run: MagicMock) -> None:
        s = m.TestSystem()

        mock_subprocess_run.return_value = CompletedProcess(
            CMD, 10, stdout=b"<output>"
        )

        result = s.run(CMD.split(), "test/file")
        mock_subprocess_run.assert_called()

        assert result.invocation == CMD
        assert result.test_display == "test/file"
        assert result.time is not None
        assert result.time > timedelta(0)
        assert not result.skipped
        assert not result.timeout
        assert result.returncode == 10
        assert result.output == "<output>"
        assert not result.passed

    def test_run_with_stdout_decode_errors(
        self, mock_subprocess_run: MagicMock
    ) -> None:
        s = m.TestSystem()

        mock_subprocess_run.return_value = CompletedProcess(
            CMD, 10, stdout=b"\xfe\xb1a"
        )

        result = s.run(CMD.split(), "test/file")
        mock_subprocess_run.assert_called()

        assert result.invocation == CMD
        assert result.test_display == "test/file"
        assert result.time is not None
        assert result.time > timedelta(0)
        assert not result.skipped
        assert not result.timeout
        assert result.returncode == 10
        assert result.output == "��a"
        assert not result.passed

    def test_dry_run(self, mock_subprocess_run: MagicMock) -> None:
        s = m.TestSystem(dry_run=True)

        result = s.run(CMD.split(), "test/file")
        mock_subprocess_run.assert_not_called()

        assert result.output == ""
        assert result.skipped

    def test_timeout(self) -> None:
        s = m.TestSystem()

        result = s.run(["sleep", "2"], "test/file", timeout=1)

        assert result.timeout
        assert not result.skipped

    def test_memort(self) -> None:
        s = m.TestSystem()

        assert s.memory == psutil.virtual_memory().total
