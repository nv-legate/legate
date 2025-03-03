#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys
from datetime import datetime
from re import IGNORECASE

from util.re_replacement import RegexReplacement, Replacement


def main() -> int:
    cur_year = datetime.now().year
    repl = Replacement(
        pattern=r"(([\d]+)\-?[\d]*)\s*(nvidia\s+corporation)",
        repl=rf"\2-{cur_year} \3",
        pragma_keyword="copyright",
        flags=IGNORECASE,
    )
    return RegexReplacement(
        description="Find and fix the date in copyright notices",
        replacements=[repl],
        allowed_suffixes=RegexReplacement.AllSuffixes,
    ).main()


if __name__ == "__main__":
    sys.exit(main())
