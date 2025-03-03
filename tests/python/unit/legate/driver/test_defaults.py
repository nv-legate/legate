# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from importlib import reload
from pathlib import Path
from typing import TYPE_CHECKING

import legate.driver.defaults as m

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def test_LEGATE_NODES() -> None:
    assert m.LEGATE_NODES == 1


def test_LEGATE_RANKS_PER_NODE() -> None:
    assert m.LEGATE_RANKS_PER_NODE == 1


def test_LEGATE_LOG_DIR(mocker: MockerFixture) -> None:
    mocker.patch("pathlib.Path.cwd", return_value=Path("foo"))
    reload(m)
    assert Path("foo") == m.LEGATE_LOG_DIR
