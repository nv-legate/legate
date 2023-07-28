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
    DEFAULT_CPUS_PER_NODE,
    DEFAULT_GPU_DELAY,
    DEFAULT_GPU_MEMORY_BUDGET,
    DEFAULT_GPUS_PER_NODE,
    DEFAULT_NUMAMEM,
    DEFAULT_OMPS_PER_NODE,
    DEFAULT_OMPTHREADS,
    DEFAULT_PROCESS_ENV,
    FEATURES,
    LAST_FAILED_FILENAME,
    PER_FILE_ARGS,
    SKIPPED_EXAMPLES,
)


class TestConsts:
    def test_DEFAULT_CPUS_PER_NODE(self) -> None:
        assert DEFAULT_CPUS_PER_NODE == 2

    def test_DEFAULT_GPUS_PER_NODE(self) -> None:
        assert DEFAULT_GPUS_PER_NODE == 1

    def test_DEFAULT_GPU_DELAY(self) -> None:
        assert DEFAULT_GPU_DELAY == 2000

    def test_DEFAULT_GPU_MEMORY_BUDGET(self) -> None:
        assert DEFAULT_GPU_MEMORY_BUDGET == 4096

    def test_DEFAULT_OMPS_PER_NODE(self) -> None:
        assert DEFAULT_OMPS_PER_NODE == 1

    def test_DEFAULT_OMPTHREADS(self) -> None:
        assert DEFAULT_OMPTHREADS == 4

    def test_DEFAULT_NUMAMEM(self) -> None:
        assert DEFAULT_NUMAMEM == 0

    def test_DEFAULT_PROCESS_ENV(self) -> None:
        assert DEFAULT_PROCESS_ENV == {
            "LEGATE_TEST": "1",
        }

    def test_FEATURES(self) -> None:
        assert FEATURES == ("cpus", "cuda", "eager", "openmp")

    def test_LAST_FAILED_FILENAME(self) -> None:
        assert LAST_FAILED_FILENAME == ".legate-test-last-failed"

    def test_SKIPPED_EXAMPLES(self) -> None:
        assert isinstance(SKIPPED_EXAMPLES, set)
        assert all(isinstance(x, str) for x in SKIPPED_EXAMPLES)
        assert all(x.startswith("examples") for x in SKIPPED_EXAMPLES)

    def test_PER_FILE_ARGS(self) -> None:
        assert isinstance(PER_FILE_ARGS, dict)
        assert all(isinstance(x, str) for x in PER_FILE_ARGS.keys())
        assert all(isinstance(x, list) for x in PER_FILE_ARGS.values())
