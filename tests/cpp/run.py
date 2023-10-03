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


def fetch_test_names(gtest_file):
    list_command = [gtest_file] + ["--gtest_list_tests"]

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
        "--gtest-file",
        dest="gtest_file",
        required=False,
        default="build/cpp_tests",
        help="GTest file under test",
    )
    parser.add_argument(
        "--mpi-rank",
        dest="mpi_rank",
        required=False,
        type=int,
        default=0,
        help="Runs mpirun with rank if non-zero",
    )
    config, extra_args = parser.parse_known_args()

    if config.mpi_rank != 0:
        extra_args += ["--mpi-rank", str(config.mpi_rank)]
        extra_args += ["--mpi-output-filename", "build/mpi_result"]
    extra_args += ["--gtest-file", config.gtest_file]
    extra_args += ["--gtest-tests"] + fetch_test_names(config.gtest_file)

    config = Config(extra_args)

    system = TestSystem(dry_run=config.dry_run)

    plan = TestPlan(config, system)

    return plan.execute()


if __name__ == "__main__":
    sys.exit(main())
