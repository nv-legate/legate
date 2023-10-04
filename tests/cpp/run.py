#!/usr/bin/env python3

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


import argparse
import subprocess
import sys

from legate.tester.config import Config
from legate.tester.test_plan import TestPlan
from legate.tester.test_system import TestSystem


def fetch_test_names(gtest_file, filter):
    list_command = [
        gtest_file,
        "--gtest_list_tests",
        f"--gtest_filter={filter}",
    ]

    result = subprocess.check_output(list_command, stderr=subprocess.STDOUT)
    result = result.decode(sys.stdout.encoding).split("\n")

    test_group = ""
    test_names = []
    for line in result:
        # Skip empty entry
        if not line.strip():
            continue

        # Check if this is a test group
        if line[0] != " ":
            test_group = line.strip()
            continue

        # Assign test to test group
        test_names += [test_group + line.strip()]

    return test_names


def main():
    parser = argparse.ArgumentParser(description="Run Legate cpp tests.")
    parser.add_argument(
        "--filter",
        default="*",
        help="Only run tests matching the GTest filter",
    )
    parser.add_argument(
        "--gtest-file",
        dest="gtest_file",
        required=False,
        default="build/cpp_tests",
        help="GTest file under test",
    )
    parser.add_argument(
        "--ranks-per-node",
        dest="ranks_per_node",
        required=False,
        type=int,
        default=1,
        help="Number of ranks per node to use (will use mpirun if > 1)",
    )
    args, extra_args = parser.parse_known_args()

    if args.ranks_per_node > 1:
        extra_args += ["--ranks-per-node", str(args.ranks_per_node)]
        extra_args += ["--mpi-output-filename", "build/mpi_result"]
    extra_args += ["--gtest-file", args.gtest_file]
    extra_args += ["--gtest-tests"] + fetch_test_names(
        args.gtest_file, args.filter
    )

    config = Config([sys.argv[0]] + extra_args)

    system = TestSystem(dry_run=config.dry_run)

    plan = TestPlan(config, system)

    return plan.execute()


if __name__ == "__main__":
    sys.exit(main())
