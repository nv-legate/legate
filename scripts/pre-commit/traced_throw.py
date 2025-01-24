#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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

import sys
from pathlib import Path
from re import Match, sub as re_sub
from typing import Final

from util.re_replacement import RegexReplacement, Replacement, ReplacementError


def is_header(path: Path) -> bool:
    return any(
        suff == path.suffix
        for suff in (".h", ".inl", ".cuh", ".cuinl", ".hpp", ".inc", ".HH")
    )


def is_publically_accessible(path: Path) -> bool:
    return "detail" not in path.parts


def _get_traced_exception_header() -> Path:
    p = Path("legate") / "utilities" / "detail" / "traced_exception.h"
    assert (
        "class TracedException"
        in (Path(__file__).parents[2] / "src" / "cpp" / p).read_text()
    )
    return p


TRACED_EXCEPTION_HEADER: Final = _get_traced_exception_header()


def update_headers(path: Path, text: str) -> str:
    if is_header(path) and is_publically_accessible(path):
        raise ReplacementError(
            path,
            "file is publically exposed, cannot include "
            f"{TRACED_EXCEPTION_HEADER} in it. Please move all throw "
            "expressions in it to a .cc",
        )
    # This assumes there is at least 1 included header, but if the user is
    # throwing a standard exception then surely they have included at least
    # <stdexcept> first right ;)? (In any case, if they haven't we add it
    # below as well).
    #
    # We unconditionally insert the headers as we let clang-format clean them
    # up after the fact. Order of the includes also doesn't matter for the
    # same reason.
    return re_sub(
        "(#include .*)",
        rf"\1\n#include <{TRACED_EXCEPTION_HEADER}>\n#include <stdexcept>",
        text,
        count=1,
    )


def is_in_comment(re_match: Match) -> bool:
    string = re_match.string
    prev_end = re_match.start()
    line_begin = string.rindex("\n", 0, prev_end) + 1
    line_prefix = string[line_begin:prev_end]
    return line_prefix.lstrip().startswith(("//", "/*", "*"))


def _make_legate_symbol(sym: str) -> tuple[str, str, str]:
    assert not sym.startswith("::")
    return (sym, f"detail::{sym}", f"legate::detail::{sym}")


IGNORED_EXCEPTION_TYPES: Final = (
    "std::bad_alloc",
    *_make_legate_symbol("TracedException"),
    *_make_legate_symbol("TracedExceptionBase"),
)


def is_ignored_exception_type(re_match: Match) -> bool:
    return re_match.group("exn_type").startswith(IGNORED_EXCEPTION_TYPES)


def wrap_throw(_path: Path, re_match: Match) -> str:
    if is_in_comment(re_match) or is_ignored_exception_type(re_match):
        return re_match[0]

    prev_text = re_match.string[: re_match.start()]
    # This is a very crude hack which breaks if someone opens and closes
    # the namespaces before the point of our match. We could of course do a
    # more exhaustive check (that the current code is still inside the
    # namespace) but:
    #
    # 1. That's a lot more complicated.
    # 2. And expensive.
    # 3. And if we're wrong below, the compiler will error out anyways.
    if "namespace legate::detail {" in prev_text:
        namespace = ""
    elif "namespace legate {" in prev_text:
        namespace = "detail::"
    else:
        namespace = "legate::detail::"
    exn_type = re_match.group("exn_type")
    return f" throw {namespace}TracedException<{exn_type}>"


def make_repl() -> list[Replacement]:
    return [
        Replacement(
            # Leading space is important. We don't want to match doxygen @throw
            pattern=r" throw\s+(?P<exn_type>[\w:_]+)",
            repl=wrap_throw,
            pragma_keyword="trace",
            on_file_change=update_headers,
        )
    ]


def main() -> int:
    return RegexReplacement(
        description="Convert throw statements to throwing TracedException",
        replacements=make_repl(),
    ).main()


if __name__ == "__main__":
    sys.exit(main())
