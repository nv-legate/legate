#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES.
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
from datetime import datetime
from pathlib import Path
from re import IGNORECASE, Match
from subprocess import check_output
from typing import Final

from util.re_replacement import RegexReplacement, Replacement

CUR_YEAR: Final = datetime.now().year

CHECKIN_YEAR_CACHE: dict[Path, str] = {}


class CopyrightReplacement(RegexReplacement):
    def parse_args(self) -> None:
        super().parse_args()

        get_all_years = Path(__file__).parent / "get_checkin_year.bash"
        assert get_all_years.is_file(), f"{get_all_years}"

        checkin_years = check_output([get_all_years, *self.files], text=True)

        items = (line.split("->") for line in checkin_years.splitlines())
        sanitized_items = (
            (path.strip(), year.strip()) for path, year in items
        )
        CHECKIN_YEAR_CACHE.update(
            {Path(path).resolve(): year for path, year in sanitized_items}
        )


def fixup_copyright(path: Path, re_match: Match) -> str:
    checkin_year = CHECKIN_YEAR_CACHE[path.resolve()]
    return f"{checkin_year}-{CUR_YEAR} {re_match[2].lstrip()}"


def main() -> int:
    repl = [
        Replacement(
            pattern=r"([\d]+\-?[\d]*)(\s+nvidia\s+corporation)",
            repl=fixup_copyright,
            pragma_keyword="copyright",
            flags=IGNORECASE,
        )
    ]
    return CopyrightReplacement(
        description="Find and fix the date in copyright notices",
        replacements=repl,
        allowed_suffixes=RegexReplacement.AllSuffixes,
    ).main()


if __name__ == "__main__":
    sys.exit(main())
