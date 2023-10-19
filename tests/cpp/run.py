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


def filter_parsed_args(action_list, parsed_args, argv):
    new_argv = []
    i = 0
    while i < len(argv):
        skip = False
        cl_arg = argv[i]
        for action in action_list:
            # may have multiple if the option allows short flags (e.g. -f and
            # -foo)
            for farg in action.option_strings:
                # use startswith to handle the argparse "allow_abbrev" option
                if not cl_arg.startswith(farg):
                    continue

                # OK we found an offender, now need to skip the flag, as
                # well as all of the values it gobbles up
                skip = True
                # skip the flag itself
                i += 1
                arg_values = getattr(parsed_args, action.dest)
                # since we don't want to skip len(some_str)
                # (which should only count as a single arg)
                if isinstance(arg_values, (list, tuple)):
                    # the flag has multiple values, skip them all
                    i += len(arg_values)
                else:
                    # only single value
                    i += 1
                # no need to check the other arg possibilities for this option
                # (if any), we already matched
                break
            if skip:
                break

        if not skip:
            new_argv.append(cl_arg)
            i += 1
    return new_argv


def fetch_test_names(gtest_file, filter):
    list_command = [
        gtest_file,
        "--gtest_list_tests",
        f"--gtest_filter={filter}",
    ]

    try:
        result = subprocess.check_output(
            list_command, stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as cpe:
        if cpe.stdout:
            print("stdout:\n" + cpe.stdout.decode())
        if cpe.stderr:
            print("stderr:\n" + cpe.stderr.decode())
        raise

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
    for action in tester_parser._actions:
        if action.dest == "gtest_file":
            # This option is required here
            action.required = True
            break
    else:
        raise RuntimeError(
            "Either the name of --gtest-file option or the value of its 'dest'"
            " attribute has changed. Could not find it in parent-parser list "
            "of options. Please update the logic here!"
        )

    parser = argparse.ArgumentParser(
        description="Run Legate cpp tests.",
        parents=[tester_parser],
        add_help=False,
    )
    filter_arg = parser.add_argument(
        "--filter",
        default="*",
        help="Only run tests matching the GTest filter",
    )
    args, extra_args = parser.parse_known_args()

    if args.ranks_per_node > 1 and not args.mpi_output_filename:
        # Let's just require this to be passed in instead of assuming the
        # location
        parser.error(
            "Must pass --mpi-output-filename if ranks per node "
            f"(have {args.ranks_per_node}) > 1"
        )
    if not args.gtest_tests:
        extra_args += ["--gtest-tests"] + fetch_test_names(
            args.gtest_file, args.filter
        )

    argv = filter_parsed_args([filter_arg], args, sys.argv)

    config = Config(argv + extra_args)

    system = TestSystem(dry_run=config.dry_run)

    plan = TestPlan(config, system)

    return plan.execute()


if __name__ == "__main__":
    sys.exit(main())
