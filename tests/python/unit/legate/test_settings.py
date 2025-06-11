# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

import legate.settings as m
from legate.util.settings import EnvOnlySetting, PrioritizedSetting

_expected_settings = ("limit_stdout",)

ENV_HEADER = (
    Path(__file__).parents[4]
    / "src"
    / "cpp"
    / "legate"
    / "utilities"
    / "detail"
    / "env_defaults.h"
).resolve(strict=True)


class TestSettings:
    def test_standard_settings(self) -> None:
        settings = [
            k
            for k, v in m.settings.__class__.__dict__.items()
            if isinstance(v, (PrioritizedSetting, EnvOnlySetting))
        ]
        assert set(settings) == set(_expected_settings)

    @pytest.mark.parametrize("name", _expected_settings)
    def test_prefix(self, name: str) -> None:
        ps = getattr(m.settings, name)
        assert ps.env_var.startswith("LEGATE_")
