# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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
