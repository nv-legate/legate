#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Final

from rich.console import Console
from rich.syntax import Syntax

from util.match_util import is_in_comment, is_in_test_dir
from util.re_replacement import RegexReplacement, Replacement, ReplacementError

if TYPE_CHECKING:
    from pathlib import Path
    from re import Match

PRAGMA_KEYWORD: Final = "switch-default"
CONSOLE: Final = Console()


def check_switch_default(path: Path, re_match: Match) -> str:
    if is_in_comment(re_match) or is_in_test_dir(path):
        # We don't care about the test directories. Well, actually we do, but
        # they have absolutely heinous abuse of switch to handle all those
        # "index" variations that trying to clean them all up would take
        # decades. So for now we demurr.
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
            "Found instance of 'default:' in switch statement at "
            f"{path}:{line_no}:"
        )
        CONSOLE.print(syntax)
        CONSOLE.print(
            "Do not use default, instead, cover all possibilities of the "
            "switch. This check does [bold red]NOT TYPE CHECK[/]. If this "
            "switch is not over an enum, please silence it by adding "
            f"'// legate-lint: no-{PRAGMA_KEYWORD}' to the end of the line."
        )
    m = cap.get()
    raise ReplacementError(path, m)


def main() -> int:
    repl = [
        Replacement(
            pattern=r"default:",
            repl=check_switch_default,
            pragma_keyword=PRAGMA_KEYWORD,
        )
    ]
    return RegexReplacement(
        description=(
            "Search for instances of default: in switch statements and "
            "ban them"
        ),
        replacements=repl,
    ).main()


if __name__ == "__main__":
    import sys

    sys.exit(main())
