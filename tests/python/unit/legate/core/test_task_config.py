# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from legate.core import LocalTaskID, TaskConfig, VariantOptions


class TestTaskConfig:
    def test_basic(self) -> None:
        task_id = LocalTaskID(1)
        config = TaskConfig(task_id)

        assert config.task_id == task_id
        assert config.variant_options is None

    def test_variant_options(self) -> None:
        task_id = LocalTaskID(1)
        options = VariantOptions(
            concurrent=True,
            has_side_effect=True,
            may_throw_exception=True,
            communicators=["foo", "bar"],
        )
        config = TaskConfig(task_id, options=options)

        assert config.task_id == task_id
        assert config.variant_options is not None
        assert config.variant_options == options

    def test_variant_options_setter(self) -> None:
        task_id = LocalTaskID(1)
        config = TaskConfig(task_id)

        assert config.task_id == task_id
        assert config.variant_options is None

        options = VariantOptions(
            concurrent=True,
            has_side_effect=True,
            may_throw_exception=True,
            communicators=["foo", "bar"],
        )
        config.variant_options = options

        assert config.variant_options == options


if __name__ == "__main__":
    import sys

    # add -s to args, we do not want pytest to capture stdout here since this
    # gobbles any C++ exceptions
    sys.exit(pytest.main([*sys.argv, "-s"]))
