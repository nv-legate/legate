# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Consolidate test configuration from command-line and environment."""

from __future__ import annotations

import pytest

import legate.tester.project as m
from legate.tester import FeatureType, defaults


def test_skipped_examples_default() -> None:
    skipped_examples = m.Project().skipped_examples()
    assert skipped_examples == set()


def test_custom_files_default() -> None:
    custom_files = m.Project().custom_files()
    assert custom_files == []


@pytest.mark.parametrize("feature", defaults.FEATURES)
def test_stage_env_default(feature: FeatureType) -> None:
    stage_env = m.Project().stage_env(feature)
    assert stage_env == {}
