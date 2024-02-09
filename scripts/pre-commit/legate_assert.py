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

import difflib
import re
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any


class Replacement:
    __slots__ = "pattern", "repl"

    def __init__(self, pattern: str, repl: str) -> None:
        self.pattern = re.compile(pattern)
        self.repl = repl

    def replace(self, text: str) -> str:
        return self.pattern.sub(self.repl, text)


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description=(
            "Find and fix occurrences of LegateAssert() guarded by "
            "LEGATE_USE_DEBUG in the provided file(s)."
        )
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
    parser.add_argument(
        "--dry-run", action="store_true", help="Whether to do a dry run"
    )
    opts = parser.parse_args()

    # convert a directory to recursive list of files, note this is a generator
    # now
    if len(opts.files) == 1 and opts.files[0].is_dir():
        opts.files = opts.files[0].rglob("*")
    return opts


def handle_file(
    opts: Namespace, file_path: Path, replacements: list[Replacement]
) -> bool:
    orig_text = file_path.read_text()
    old_text = orig_text
    changed = False
    for repl in replacements:
        new_text = repl.replace(old_text)
        if new_text != old_text:
            changed = True
            old_text = new_text

    if changed:
        if opts.verbose:
            diff = difflib.unified_diff(
                orig_text.splitlines(),
                new_text.splitlines(),
                fromfile="before",
                tofile="after",
            )
            diff = list(diff)[3:]
            print("\n".join(diff))
        if not opts.dry_run:
            file_path.write_text(new_text)
    return changed


def main() -> int:
    opts = parse_args()

    def vprint(*args: Any, **kwargs: Any) -> None:
        if opts.verbose:
            print(*args, **kwargs)

    repl = [
        Replacement(r"(\s+)assert\(", r"\1LegateCheck("),
        Replacement(
            r"if\s+\(LegateDefined\(LEGATE_USE_DEBUG\)\)\s+{"
            r"\s+LegateAssert\(([^;]*)\);\s+"
            r"}",
            r"LegateAssert(\1);",
        ),
        Replacement(
            r"if\s+\(LegateDefined\(LEGATE_USE_DEBUG\)\)\s+{"
            r"\s+LegateCheck\(([^;]*)\);\s+"
            r"}",
            r"LegateCheck(\1);",
        ),
    ]

    modified = []
    allowed_suff = {".h", ".hpp", ".inl", ".cc", ".cpp", ".cu", ".cuh"}
    for file_path in opts.files:
        if (file_path.suffix not in allowed_suff) or (not file_path.is_file()):
            vprint("skipping:     ", file_path)
            continue
        vprint("processing:   ", file_path)
        if handle_file(opts, file_path, repl):
            vprint("found changes:", file_path)
            modified.append(file_path)

    mod_str = "would have modified:" if opts.dry_run else "modified:"
    for file_path in modified:
        print(mod_str, file_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
