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

"""Provide an argparse ArgumentParser for the test runner.

"""
from __future__ import annotations

from argparse import ArgumentParser
from typing import Literal, Union

from typing_extensions import TypeAlias

from ..util.args import ExtendAction, MultipleChoices
from . import FEATURES, defaults

PinOptionsType: TypeAlias = Union[
    Literal["partial"],
    Literal["none"],
    Literal["strict"],
]

PIN_OPTIONS: tuple[PinOptionsType, ...] = (
    "partial",
    "none",
    "strict",
)


#: The argument parser for test.py
parser = ArgumentParser(
    description="Run the Cunumeric test suite",
    epilog="Any extra arguments will be forwarded to the Legate script",
)

stages = parser.add_argument_group("Feature stage selection")

stages.add_argument(
    "--use",
    dest="features",
    action=ExtendAction,
    choices=MultipleChoices(sorted(FEATURES)),
    type=lambda s: s.split(","),  # type: ignore [arg-type,return-value]
    help="Test this library with features (also via USE_*)",
)

selection = parser.add_argument_group("Test file selection")

selection.add_argument(
    "--files",
    nargs="+",
    default=None,
    help="Explicit list of test files to run",
)

selection.add_argument(
    "--unit",
    dest="unit",
    action="store_true",
    default=False,
    help="Include unit tests",
)

selection.add_argument(
    "-C",
    "--directory",
    dest="test_root",
    metavar="DIR",
    action="store",
    default=None,
    required=False,
    help="Root directory containing the tests subdirectory",
)

selection.add_argument(
    "--last-failed",
    action="store_true",
    default=False,
    help="Only run the failed tests from the last run",
)

selection.add_argument(
    "--gtest-file",
    dest="gtest_file",
    default=None,
    help="Path to GTest binary",
)

gtest_group = selection.add_mutually_exclusive_group()

gtest_group.add_argument(
    "--gtest-tests",
    dest="gtest_tests",
    nargs="*",
    default=[],
    help="List of GTest tests to run",
)


gtest_group.add_argument(
    "--gtest-filter",
    dest="gtest_filter",
    default=None,
    help="Pattern to filter GTest tests",
)


selection.add_argument(
    "--gtest-skip-list",
    dest="gtest_skip_list",
    nargs="*",
    default=[],
    help="List of GTest tests to skip",
)


# -- core

core = parser.add_argument_group("Core allocation")

core.add_argument(
    "--cpus",
    dest="cpus",
    type=int,
    default=defaults.CPUS_PER_NODE,
    help="Number of CPUs per node to use",
)

core.add_argument(
    "--gpus",
    dest="gpus",
    type=int,
    default=defaults.GPUS_PER_NODE,
    help="Number of GPUs per node to use",
)

core.add_argument(
    "--omps",
    dest="omps",
    type=int,
    default=defaults.OMPS_PER_NODE,
    help="Number of OpenMP processors per node to use",
)

core.add_argument(
    "--ompthreads",
    dest="ompthreads",
    metavar="THREADS",
    type=int,
    default=defaults.OMPTHREADS,
    help="Number of threads per OpenMP processor",
)

core.add_argument(
    "--utility",
    dest="utility",
    type=int,
    default=1,
    help="Number of utility CPUs to reserve for runtime services",
)

# -- memory

memory = parser.add_argument_group("Memory allocation")

memory.add_argument(
    "--sysmem",
    dest="sysmem",
    type=int,
    default=defaults.SYS_MEMORY_BUDGET,
    help="per-process CPU system memory limit (MB)",
)

memory.add_argument(
    "--fbmem",
    dest="fbmem",
    type=int,
    default=defaults.GPU_MEMORY_BUDGET,
    help="per-process GPU framebuffer memory limit (MB)",
)

memory.add_argument(
    "--numamem",
    dest="numamem",
    type=int,
    default=defaults.NUMA_MEMORY_BUDGET,
    help="per-process NUMA memory for OpenMP processors limit (MB)",
)

# -- multi_node

multi_node = parser.add_argument_group("Multi-node configuration")

multi_node.add_argument(
    "--nodes",
    dest="nodes",
    type=int,
    default=defaults.NODES,
    help="Number of nodes to use",
)

multi_node.add_argument(
    "--ranks-per-node",
    dest="ranks_per_node",
    type=int,
    default=defaults.RANKS_PER_NODE,
    help="Number of ranks per node to use",
)

multi_node.add_argument(
    "--launcher",
    dest="launcher",
    choices=["mpirun", "jsrun", "srun", "none"],
    default="none",
    help='launcher program to use (set to "none" for local runs, or if '
    "the launch has already happened by the time legate is invoked)",
)

multi_node.add_argument(
    "--launcher-extra",
    dest="launcher_extra",
    action="append",
    default=[],
    required=False,
    help="additional argument to pass to the launcher (can appear more "
    "than once)",
)

multi_node.add_argument(
    "--mpi-output-filename",
    dest="mpi_output_filename",
    default=None,
    help="Directory to dump mpirun output",
)

# -- execution

execution = parser.add_argument_group("Test execution")

execution.add_argument(
    "-j",
    "--workers",
    dest="workers",
    type=int,
    default=None,
    help="Number of parallel workers for testing",
)

execution.add_argument(
    "--timeout",
    dest="timeout",
    type=int,
    action="store",
    default=None,
    required=False,
    help="Timeout in seconds for individual tests",
)

execution.add_argument(
    "--cpu-pin",
    dest="cpu_pin",
    choices=PIN_OPTIONS,
    default="partial",
    help="CPU pinning behavior on platforms that support CPU pinning",
)

execution.add_argument(
    "--gpu-delay",
    dest="gpu_delay",
    type=int,
    default=defaults.GPU_DELAY,
    help="Delay to introduce between GPU tests (ms)",
)

execution.add_argument(
    "--bloat-factor",
    dest="bloat_factor",
    type=int,
    default=defaults.GPU_BLOAT_FACTOR,
    help="Fudge factor to adjust memory reservations",
)

# -- info

info = parser.add_argument_group("Informational")

info.add_argument(
    "-v",
    "--verbose",
    dest="verbose",
    action="count",
    default=0,
    help="Display verbose output. Use -vv for even more output (test stdout)",
)

info.add_argument(
    "--debug",
    dest="debug",
    action="store_true",
    help="Print out the commands that are to be executed",
)

# -- other

other = parser.add_argument_group("Other options")

other.add_argument(
    "--legate",
    dest="legate_dir",
    metavar="LEGATE_DIR",
    action="store",
    default=None,
    required=False,
    help="Path to Legate installation directory",
)

other.add_argument(
    "--cov-bin",
    default=None,
    help=(
        "coverage binary location, "
        "e.g. /conda_path/envs/env_name/bin/coverage"
    ),
)

other.add_argument(
    "--cov-args",
    default="run -a --branch",
    help="coverage run command arguments, e.g. run -a --branch",
)

other.add_argument(
    "--cov-src-path",
    default=None,
    help=(
        "path value of --source in coverage run command, "
        "e.g. /project_path/cunumeric/cunumeric"
    ),
)

other.add_argument(
    "--dry-run",
    dest="dry_run",
    action="store_true",
    help="Print the test plan but don't run anything",
)

other.add_argument(
    "--color",
    dest="color",
    action="store_true",
    required=False,
    help="Whether to use color terminal output (if colorama is installed)",
)
