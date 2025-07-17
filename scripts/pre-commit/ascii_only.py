#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import re
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Final

from rich.console import Console
from rich.syntax import Syntax

CONSOLE: Final = Console(stderr=True)

IGNORE_PATTERN: Final = re.compile(r"legate\-lint:.*no\-ascii\-only")


class Options:
    def __init__(self, ns: Namespace) -> None:
        self.verbose: bool = ns.verbose

        files: list[Path] = ns.files
        # convert a directory to recursive list of files
        if len(files) == 1 and files[0].is_dir():
            files = list(files[0].rglob("*"))

        self.files = files

    def vprint(self, *args: Any, **kwargs: Any) -> None:
        if self.verbose:
            print(*args, **kwargs)  # noqa: T201


def parse_args() -> Options:
    parser = ArgumentParser(
        description="Check that a file contains only ASCII characters"
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help=(
            "Path(s) to the file(s) or directories to be searched. "
            "Directories are searched recursively"
        ),
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output"
    )
    args = parser.parse_args()

    return Options(args)


def check_non_ascii(path: Path) -> bool:
    try:
        txt = path.read_text()
    except:
        print("Failed to read", path)  # noqa: T201
        raise
    # Fast check because we assume everything is OK.
    if txt.isascii():
        return False  # no error

    lexer = Syntax.guess_lexer(path)

    def print_non_ascii_lines(lines: list[int]) -> list[int]:
        if not lines:
            return lines

        syntax = Syntax(
            txt,
            lexer=lexer,
            line_range=(max(lines[0] - 3, 1), lines[-1] + 3),
            highlight_lines=set(lines),
            line_numbers=True,
            background_color="default",
        )

        CONSOLE.print(
            "Found instance of non-ASCII characters in "
            f"{path}:{lines[0]}-{lines[-1]}:"
        )
        CONSOLE.print(syntax)
        return []

    # Now do slow line-by-line search
    non_ascii_lines = []
    found_error = False
    for line_no, line in enumerate(txt.splitlines(), start=1):
        if line.isascii():
            non_ascii_lines = print_non_ascii_lines(non_ascii_lines)
            continue

        if IGNORE_PATTERN.search(line):
            continue

        non_ascii_lines.append(line_no)
        found_error = True

    # Once more in case the last lines of the file contained non-ascii chars
    print_non_ascii_lines(non_ascii_lines)
    return found_error


def main() -> int:
    opts = parse_args()
    retcode = 0

    for f in opts.files:
        if not f.is_file():
            opts.vprint("skipping:  ", f)
            continue

        opts.vprint("processing:", f)
        if check_non_ascii(f):
            retcode = 1
    return retcode


if __name__ == "__main__":
    sys.exit(main())
