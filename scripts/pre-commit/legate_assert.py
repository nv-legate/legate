#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys

from util.re_replacement import RegexReplacement, Replacement


def main() -> int:
    repl = [
        Replacement(
            r"(\s+)assert\(", r"\1LEGATE_CHECK(", pragma_keyword="assert"
        ),
        Replacement(
            r"if\s+\(LEGATE_DEFINED\(LEGATE_USE_DEBUG\)\)\s+{"
            r"\s+LEGATE_ASSERT\(([^;]*)\);\s+"
            r"}",
            r"LEGATE_ASSERT(\1);",
            pragma_keyword="assert",
        ),
        Replacement(
            r"if\s+\(LEGATE_DEFINED\(LEGATE_USE_DEBUG\)\)\s+{"
            r"\s+LEGATE_CHECK\(([^;]*)\);\s+"
            r"}",
            r"LEGATE_CHECK(\1);",
            pragma_keyword="assert",
        ),
    ]
    return RegexReplacement(
        description=(
            "Find and fix occurrences of LEGATE_ASSERT() guarded by "
            "LEGATE_USE_DEBUG in the provided file(s)."
        ),
        replacements=repl,
    ).main()


if __name__ == "__main__":
    sys.exit(main())
