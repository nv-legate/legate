# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Consolidate test configuration from command-line and environment."""

from __future__ import annotations

import re
from importlib import reload
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

import legate


def test___version___override(monkeypatch: pytest.MonkeyPatch) -> None:
    global legate  # noqa: PLW0603
    monkeypatch.setenv("LEGATE_USE_VERSION", "24.01.00")
    legate = reload(legate)
    assert legate.__version__ == "24.01.00"


def test___version___format() -> None:
    global legate  # noqa: PLW0603
    legate = reload(legate)

    # just being cautious, if the test are functioning properly, the
    # actual non-overriden version should never equal the bogus version
    # from test___version___override above
    assert legate.__version__ != "24.01.00"

    assert re.match(r"^\d{2}\.\d{2}\.\d{2}$", legate.__version__[:8])
