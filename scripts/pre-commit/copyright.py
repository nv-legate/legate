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
