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

"""Consolidate test configuration from command-line and environment."""

from __future__ import annotations

import sys

from legate.tester.defaults import (
    CPU_PIN,
    CPUS_PER_NODE,
    FEATURES,
    GPU_BLOAT_FACTOR,
    GPU_DELAY,
    GPU_MEMORY_BUDGET,
    GPUS_PER_NODE,
    NODES,
    NUMA_MEMORY_BUDGET,
    OMPS_PER_NODE,
    OMPTHREADS,
    PROCESS_ENV,
    SMALL_SYSMEM,
    SYS_MEMORY_BUDGET,
)


def test_FEATURES() -> None:
    assert FEATURES == ("cpus", "cuda", "eager", "openmp")


def test_CPUS_PER_NODE() -> None:
    assert CPUS_PER_NODE == 2


def test_GPUS_PER_NODE() -> None:
    assert GPUS_PER_NODE == 1


def test_GPU_BLOAT_FACTOR() -> None:
    assert GPU_BLOAT_FACTOR == 1.5


def test_GPU_DELAY() -> None:
    assert GPU_DELAY == 2000


def test_GPU_MEMORY_BUDGET() -> None:
    assert GPU_MEMORY_BUDGET == 4096


def test_SYS_MEMORY_BUDGET() -> None:
    assert SYS_MEMORY_BUDGET == 4000


def test_OMPS_PER_NODE() -> None:
    assert OMPS_PER_NODE == 1


def test_OMPTHREADS() -> None:
    assert OMPTHREADS == 4


def test_NUMA_MEMORY_BUDGET() -> None:
    assert NUMA_MEMORY_BUDGET == 4000


def test_SMALL_SYSMEM() -> None:
    assert SMALL_SYSMEM == 300


def test_PROCESS_ENV() -> None:
    assert PROCESS_ENV == {
        "LEGATE_TEST": "1",
        "LEGATE_CONSENSUS": "1",
        "LEGION_DEFAULT_ARGS": "-logfile stderr",
    }


def test_NODES() -> None:
    assert NODES == 1


def test_CPU_PIN() -> None:
    if sys.platform == "darwin":
        assert CPU_PIN == "none"
    else:
        assert CPU_PIN == "partial"
