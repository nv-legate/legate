# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
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

import pytest

from legate.core import is_running_in_task
from legate.core.task import task


class TestIsRunningInTask:
    def test_toplevel(self) -> None:
        assert not is_running_in_task()

    def test_in_task(self) -> None:
        @task(throws_exception=True)
        def tester() -> None:
            assert is_running_in_task()

        tester()


if __name__ == "__main__":
    sys.exit(pytest.main())
