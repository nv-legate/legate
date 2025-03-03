# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys
import textwrap
from argparse import REMAINDER, ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path

KNOWN_PATCHES = {"numpy": "cupynumeric"}

newline = "\n"

DESCRIPTION = textwrap.dedent(
    f"""
Patch existing libraries with legate equivalents.

Currently the following patching can be applied:

{newline.join(f"    {key} -> {value}" for key, value in KNOWN_PATCHES.items())}

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


def do_patch(name: str, *, verbose: bool = False) -> None:  # noqa: D103
    if name not in KNOWN_PATCHES:
        msg = f"No patch available for module {name}"
        raise ValueError(msg)

    cuname = KNOWN_PATCHES[name]
    try:
        module = __import__(cuname)
        sys.modules[name] = module
        if verbose:
            print(f"lgpatch: patched {name} -> {cuname}")  # noqa: T201
    except ImportError:
        msg = f"Could not import patch module {cuname}"
        raise RuntimeError(msg)


def main() -> None:  # noqa: D103
    args = parser.parse_args()

    if len(args.prog) == 0:
        parser.print_usage()
        sys.exit()

    if len(args.patch) == 0:
        print(  # noqa: T201
            "WARNING: lgpatch called without any --patch options"
        )

    for name in set(args.patch):
        do_patch(name, verbose=args.verbose)

    sys.argv[:] = args.prog

    with Path(args.prog[0]).open() as f:
        exec(f.read(), {"__name__": "__main__"})  # noqa: S102


if __name__ == "__main__":
    main()
