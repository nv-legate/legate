#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Final

from rich.console import Console
from rich.syntax import Syntax

from util.match_util import is_in_comment
from util.re_replacement import RegexReplacement, Replacement, ReplacementError

if TYPE_CHECKING:
    from pathlib import Path
    from re import Match


PRAGMA_KEYWORD: Final = "legate-defined"
CONSOLE: Final = Console()

KNOWN_IGNORES: Final = {
    "LEGATE_MAX_DIM",
    "LEGATE_CUDA_VERSION",
    "LEGATE_CPP_VERSION",
    "LEGATE_CPP_MIN_VERSION",
}


def is_allowed_ignore(re_match: Match) -> bool:
    return re_match[1] in KNOWN_IGNORES


def check_legate_defined(path: Path, re_match: Match) -> str:
    if is_in_comment(re_match) or is_allowed_ignore(re_match):
        return re_match[0]

    line_no = re_match.string.count("\n", 0, re_match.end()) + 1
    syntax = Syntax(
        re_match.string,
        lexer=Syntax.guess_lexer(path, re_match.string),
        line_range=(max(line_no - 3, 1), line_no + 3),
        highlight_lines={line_no},
        line_numbers=True,
        background_color="default",
    )
    with CONSOLE.capture() as cap:
        CONSOLE.print(
            "Instances of preprocessor ifdef/ifndef/if defined found at "
            f"{path}:{line_no}. Use LEGATE_DEFINED() instead:"
        )
        CONSOLE.print(syntax)
        CONSOLE.print(
            "If this is intentional, please silence it by adding "
            f"'// legate-lint: no-{PRAGMA_KEYWORD}' to the end of the line."
        )
    m = cap.get()
    raise ReplacementError(path, m)


def main() -> int:
    repl = [
        Replacement(
            pattern=r"#\s*(?:if|else\s+if|elif).*(?<!LEGATE_DEFINED\()(LEGATE_(?!DEFINED)[\w_\d]+)",
            repl=check_legate_defined,
            pragma_keyword=PRAGMA_KEYWORD,
        )
    ]
    return RegexReplacement(
        description=(
            "Find uses of LEGATE_ macros not expanded with LEGATE_DEFINED()"
        ),
        replacements=repl,
    ).main()


if __name__ == "__main__":
    sys.exit(main())
