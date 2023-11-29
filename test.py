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

# Skip these tests in mutli-node testing, as they don't pass with MPI; for some
# reason the abort calls in these tests are not completely neutralized by Gtest
# in an MPI environment.
SKIP_LIST = {
    "AttachDeathTest.MissingManualDetach",
    "DeathTestExample.Simple",
}


def main() -> int:
    for action in tester_parser._actions:
        match action.dest:
            case "gtest_file":
                action.default = os.path.abspath(
                    "./build/debug-gcc/legate-core-cpp/tests/cpp/bin/cpp_tests"
                )
            case "mpi_output_filename":
                action.default = os.path.abspath(
                    "./build/debug-gcc/legate-core-cpp/tests/cpp/mpi_result"
                )

    config = Config(sys.argv)

    system = TestSystem(dry_run=config.dry_run)

    plan = TestPlan(config, system)

    return plan.execute()


if __name__ == "__main__":
    sys.exit(main())
