# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.B
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest

from ..logger import Logger

if TYPE_CHECKING:
    from pathlib import Path


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
        assert isinstance(logger._file_logger, logging.Logger)
        assert logger.file_path == tmp_configure_log
        assert len(logger._row_data) == 0
        assert not logger._live.is_started

    def test_flush(self, logger: Logger) -> None:
        logger.flush()

    @pytest.mark.parametrize("mess", ("hello world", "goodbye world"))
    def test_log_screen(
        self, logger: Logger, capsys: pytest.CaptureFixture[str], mess: str
    ) -> None:
        with logger:
            logger.log_screen(mess=mess)
        captured = capsys.readouterr()
        assert mess in captured.out
        assert captured.err == ""

    def test_logger_context(self, logger: Logger) -> None:
        assert logger._live.is_started is False
        with logger as lg:
            # Need to use alias lg since otherwise mypy says the final line is
            # unreachable, since I guess it assumes the lifetime of the
            # variable "logger" is tied to the with statement?
            assert lg is logger
            assert lg._live.is_started is True
        assert logger._live.is_started is False

    def test_append_live_message(self, logger: Logger) -> None:
        row_data = logger._row_data
        # make sure that the above access doesn't return a copy or something
        # like that
        assert row_data is logger._row_data
        assert row_data.maxlen is not None
        assert len(row_data) == 0
        logger._append_live_message("foo", keep=True)
        assert len(row_data) == 1
        assert row_data[0] == ("foo", True)
        for i in range(row_data.maxlen - len(row_data)):
            row_data.append((f"bar_{i}", False))
        assert len(row_data) == row_data.maxlen
        assert row_data[0] == ("foo", True)
        logger._append_live_message("new_foo", keep=True)
        assert len(row_data) == row_data.maxlen
        assert row_data[0] == ("foo", True)
        # The last non-kept entry should now be next
        assert row_data[1] == ("bar_1", False)

    def test_append_live_message_full(self, logger: Logger) -> None:
        assert logger._row_data.maxlen is not None
        for i in range(logger._row_data.maxlen):
            logger._row_data.append((f"foo_{i}", True))
        with pytest.raises(
            ValueError,
            match=(
                "Could not prune row data, every entry was marked as "
                "persistent"
            ),
        ):
            logger._append_live_message("oh no", keep=True)

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
