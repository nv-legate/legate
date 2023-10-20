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


import sys

try:
    # imported for effect
    import legate  # noqa F401
except ModuleNotFoundError:
    import os

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    legate_dir = os.path.abspath(
        os.path.join(cur_dir, os.path.pardir, os.path.pardir)
    )
    sys.path.append(legate_dir)

    del cur_dir
    del legate_dir

from legate.tester.args import parser as tester_parser
from legate.tester.config import Config
from legate.tester.test_plan import TestPlan
from legate.tester.test_system import TestSystem

# Skip these tests in mutli-node testing, as they don't pass with MPI; for some
# reason the abort calls in these tests are not completely neutralized by Gtest
# in an MPI environment.
SKIP_LIST = {
    "AttachDeathTest.MissingManualDetach",
    "DeathTestExample.Simple",
}


def main():
    for action in tester_parser._actions:
        if action.dest == "gtest_file":
            action.default = "build/cpp_tests"
        elif action.dest == "mpi_output_filename":
            action.default = "build/mpi_result"

    config = Config(sys.argv)

    system = TestSystem(dry_run=config.dry_run)

    plan = TestPlan(config, system)

    return plan.execute()


if __name__ == "__main__":
    sys.exit(main())
