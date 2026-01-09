# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from legate.core import ResourceConfig


class TestResourceConfig:
    def test_basic(self) -> None:
        rc = ResourceConfig()

        assert rc.max_tasks == 1024
        assert rc.max_dyn_tasks == 0
        assert rc.max_reduction_ops == 0
        assert rc.max_projections == 0
        assert rc.max_shardings == 0

    def test_construct(self) -> None:
        max_tasks = 123
        max_dyn_tasks = 42
        max_reduction_ops = 456
        max_projections = 22
        max_shardings = 88

        rc = ResourceConfig(
            max_tasks=max_tasks,
            max_dyn_tasks=max_dyn_tasks,
            max_reduction_ops=max_reduction_ops,
            max_projections=max_projections,
            max_shardings=max_shardings,
        )

        assert rc.max_tasks == max_tasks
        assert rc.max_dyn_tasks == max_dyn_tasks
        assert rc.max_reduction_ops == max_reduction_ops
        assert rc.max_projections == max_projections
        assert rc.max_shardings == max_shardings

    def test_setters(self) -> None:
        rc = ResourceConfig()

        max_tasks = 123
        max_dyn_tasks = 42
        max_reduction_ops = 456
        max_projections = 22
        max_shardings = 88

        rc.max_tasks = max_tasks
        rc.max_dyn_tasks = max_dyn_tasks
        rc.max_reduction_ops = max_reduction_ops
        rc.max_projections = max_projections
        rc.max_shardings = max_shardings

        assert rc.max_tasks == max_tasks
        assert rc.max_dyn_tasks == max_dyn_tasks
        assert rc.max_reduction_ops == max_reduction_ops
        assert rc.max_projections == max_projections
        assert rc.max_shardings == max_shardings


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
