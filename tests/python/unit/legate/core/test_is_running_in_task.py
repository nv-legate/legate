# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys

import pytest

from legate.core import VariantOptions, is_running_in_task
from legate.core.task import task


class TestIsRunningInTask:
    def test_toplevel(self) -> None:
        assert not is_running_in_task()

    def test_in_task(self) -> None:
        @task(options=VariantOptions(may_throw_exception=True))
        def tester() -> None:
            assert is_running_in_task()

        tester()


if __name__ == "__main__":
    sys.exit(pytest.main())
