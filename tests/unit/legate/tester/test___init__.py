# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Consolidate test configuration from command-line and environment.

"""
from __future__ import annotations

from legate.tester import (
    FEATURES,
    LAST_FAILED_FILENAME,
    PER_FILE_ARGS,
    SKIPPED_EXAMPLES,
)


def test_FEATURES() -> None:
    assert FEATURES == ("cpus", "cuda", "eager", "openmp")


def test_LAST_FAILED_FILENAME() -> None:
    assert LAST_FAILED_FILENAME == ".legate-test-last-failed"


def test_SKIPPED_EXAMPLES() -> None:
    assert isinstance(SKIPPED_EXAMPLES, set)
    assert all(isinstance(x, str) for x in SKIPPED_EXAMPLES)
    assert all(x.startswith("examples") for x in SKIPPED_EXAMPLES)


def test_PER_FILE_ARGS() -> None:
    assert isinstance(PER_FILE_ARGS, dict)
    assert all(isinstance(x, str) for x in PER_FILE_ARGS.keys())
    assert all(isinstance(x, list) for x in PER_FILE_ARGS.values())
