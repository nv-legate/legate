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
from __future__ import annotations

import os
import sys

# Since this file sits top-level, a simple "import legate" would ALWAYS import
# pwd/legate/__init__.py. This behavior is not desirable/surprising if the user
# has already legate.core.
#
# To get around this, we move the current directory to the end of the module
# search path. That way the global modules are searched first, and the legate
# directory does not shadow the installed version.
del sys.path[0]
sys.path.append("")

# These are all still "top of file", but flake8 is feeling feisty about it
from legate.tester.args import parser as tester_parser  # noqa E402
from legate.tester.config import Config  # noqa E402
from legate.tester.test_plan import TestPlan  # noqa E402
from legate.tester.test_system import TestSystem  # noqa E402

BUILD_DIR = "./build"


def find_latest_cpp_test_dir() -> str | None:
    if not os.path.exists(BUILD_DIR):
        return None

    # Find the build directory that is updated the latest
    build_dirs = sorted(
        filter(lambda dir: dir.is_dir(), os.scandir(BUILD_DIR)),
        key=lambda dir: -dir.stat().st_mtime,
    )

    def _make_test_dir(prefix: os.DirEntry) -> str:
        return os.path.join(prefix.path, "legate-core-cpp", "tests", "cpp")

    if len(build_dirs) == 0 or not os.path.exists(
        test_dir := os.path.abspath(_make_test_dir(build_dirs[0]))
    ):
        return None

    return test_dir


def main() -> int:
    if (test_dir := find_latest_cpp_test_dir()) is not None:
        for action in tester_parser._actions:
            match action.dest:
                case "gtest_file":
                    action.default = os.path.join(test_dir, "bin", "cpp_tests")
                case "mpi_output_filename":
                    action.default = os.path.join(test_dir, "mpi_result")

    config = Config(sys.argv)

    system = TestSystem(dry_run=config.dry_run)

    plan = TestPlan(config, system)

    return plan.execute()


if __name__ == "__main__":
    sys.exit(main())
