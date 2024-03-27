# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.B
#                         All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pytest import LogCaptureFixture

from ..logger import Logger


@pytest.fixture
def tmp_configure_log(AEDIFIX_PYTEST_DIR: Path) -> Path:
    return AEDIFIX_PYTEST_DIR / "configure.log"


@pytest.fixture
def logger(tmp_configure_log: Path) -> Logger:
    return Logger(tmp_configure_log)


class TestLogger:
    def test_create(self, tmp_configure_log: Path) -> None:
        import logging

        logger = Logger(tmp_configure_log)
        assert isinstance(logger._screen_logger, logging.Logger)
        assert isinstance(logger._file_logger, logging.Logger)
        assert logger.file_path == tmp_configure_log

    def test_flush(self, logger: Logger) -> None:
        logger.flush()

    @pytest.mark.parametrize("mess", ("hello world", "goodbye world"))
    @pytest.mark.parametrize("end", ("\n", "\r", "!!!"))
    @pytest.mark.parametrize("flush", (True, False))
    def test_log_screen(
        self,
        logger: Logger,
        caplog: LogCaptureFixture,
        mess: str,
        end: str,
        flush: bool,
    ) -> None:
        logger.log_screen(mess=mess, end=end, flush=flush)
        assert len(caplog.messages) == 1
        expected = mess + ("" if end in {"\n", "\r"} else end)
        assert caplog.messages[0] == expected

    def test_log_screen_clear_line(
        self, logger: Logger, caplog: LogCaptureFixture
    ) -> None:
        logger.log_screen_clear_line()
        assert len(caplog.messages) == 1
        assert caplog.messages[0] == "\r\033[K"

    def test_log_file(self, logger: Logger) -> None:
        mess = "foo bar baz"
        logger.log_file(mess)
        assert logger.file_path.read_text() == mess + "\n"

        mess2 = "asdasdasdasd qwdoiqnwdnqwid\ndqowdqowdqiwodqowdi"
        logger.log_file(mess2)
        assert logger.file_path.read_text() == mess + "\n" + mess2 + "\n"

    def test_copy_log(self, logger: Logger) -> None:
        mess = "foo, bar, baz"
        logger.log_file(mess)
        orig_log = logger.file_path
        other_log = orig_log.parent / "backup_log.log"
        assert not other_log.exists()
        dest = logger.copy_log(other_log)
        full_mess = (
            f"{mess}\nCopying file log from {orig_log} to {other_log}\n"
        )
        assert dest == other_log
        assert other_log.exists()
        assert other_log.is_file()
        assert other_log.read_text() == full_mess
        assert orig_log.exists()
        assert orig_log.is_file()
        assert orig_log.read_text() == full_mess


if __name__ == "__main__":
    sys.exit(pytest.main())
