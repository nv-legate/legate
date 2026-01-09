# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Consolidate test configuration from command-line and environment."""

from __future__ import annotations

from legate.tester import LAST_FAILED_FILENAME


def test_LAST_FAILED_FILENAME() -> None:
    assert LAST_FAILED_FILENAME.endswith(".legate-test-last-failed")
