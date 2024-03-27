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
from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path
from typing import Any


class Replacement:
    __slots__ = "pattern", "repl"

    def __init__(self, pattern: str, repl: str, flags: int = 0) -> None:
        self.pattern = re.compile(pattern, flags=flags)
        self.repl = repl

    def replace(self, text: str) -> str:
        return self.pattern.sub(self.repl, text)


class RegexReplacement:
    def __init__(
        self, description: str, replacements: Sequence[Replacement]
    ) -> None:
        assert len(replacements)
        self.description = description
        self.replacements = replacements
        self.files = []
        self.dry_run = False
        self.verbose = False

    def parse_args(self) -> None:
        parser = ArgumentParser(description=self.description)
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

        # convert a directory to recursive list of files, note this is a
        # generator now
        if len(opts.files) == 1 and opts.files[0].is_dir():
            opts.files = opts.files[0].rglob("*")

        self.files = opts.files
        self.dry_run = opts.dry_run
        self.verbose = opts.verbose

    def handle_file(self, file_path: Path) -> bool:
        orig_text = file_path.read_text()
        old_text = orig_text
        changed = False
        for repl in self.replacements:
            new_text = repl.replace(old_text)
            if new_text != old_text:
                changed = True
                old_text = new_text

        if changed:
            if self.verbose:
                diff = difflib.unified_diff(
                    orig_text.splitlines(),
                    new_text.splitlines(),
                    fromfile="before",
                    tofile="after",
                )
                diff = list(diff)[3:]
                print("\n".join(diff))
            if not self.dry_run:
                file_path.write_text(new_text)
        return changed

    def vprint(self, *args: Any, **kwargs: Any) -> None:
        if self.verbose:
            print(*args, **kwargs)

    def main(self) -> int:
        self.parse_args()

        modified = []
        allowed_suff = {".h", ".hpp", ".inl", ".cc", ".cpp", ".cu", ".cuh"}
        for file_path in self.files:
            if (file_path.suffix not in allowed_suff) or (
                not file_path.is_file()
            ):
                self.vprint("skipping:     ", file_path)
                continue
            self.vprint("processing:   ", file_path)
            if self.handle_file(file_path):
                self.vprint("found changes:", file_path)
                modified.append(file_path)

        mod_str = "would have modified:" if self.dry_run else "modified:"
        for file_path in modified:
            print(mod_str, file_path)
        return 0
