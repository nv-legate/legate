#! /usr/bin/env legate
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import sys
import textwrap
from argparse import REMAINDER, ArgumentParser, RawDescriptionHelpFormatter

KNOWN_PATCHES = {"numpy": "cunumeric"}

newline = "\n"

DESCRIPTION = textwrap.dedent(
    f"""
Patch existing libraries with legate equivalents.

Currently the following patching can be applied:

{newline.join(f'    {key} -> {value}' for key, value in KNOWN_PATCHES.items())}

"""
)

EPILOG = """
Any additional command line arguments are passed on to PROG as-is
"""


parser = ArgumentParser(
    prog="lgpatch",
    description=DESCRIPTION,
    allow_abbrev=False,
    add_help=True,
    epilog=EPILOG,
    formatter_class=RawDescriptionHelpFormatter,
)
parser.add_argument(
    "prog",
    metavar="PROG",
    nargs=REMAINDER,
    help="The legate program (with any arguments) to run",
)
parser.add_argument(
    "-p",
    "--patch",
    action="append",
    help="Patch the specified libraries. (May be supplied multiple times)",
    default=[],
)
parser.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    help="print out more verbose information about patching",
    default=False,
)


def do_patch(name: str, verbose: bool = False) -> None:
    if name not in KNOWN_PATCHES:
        raise ValueError(f"No patch available for module {name}")

    cuname = KNOWN_PATCHES[name]
    try:
        module = __import__(cuname)
        sys.modules[name] = module
        if verbose:
            print(f"lgpatch: patched {name} -> {cuname}")
    except ImportError:
        raise RuntimeError(f"Could not import patch module {cuname}")


def main() -> None:
    args = parser.parse_args()

    if len(args.prog) == 0:
        parser.print_usage()
        sys.exit()

    if len(args.patch) == 0:
        print("WARNING: lgpatch called without any --patch options")

    for name in set(args.patch):
        do_patch(name, args.verbose)

    sys.argv[:] = args.prog

    with open(args.prog[0]) as f:
        exec(f.read(), {"__name__": "__main__"})


if __name__ == "__main__":
    main()
