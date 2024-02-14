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

import sys

from util.re_replacement import RegexReplacement, Replacement


def main() -> int:
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
    return RegexReplacement(
        description=(
            "Find and fix occurrences of LegateAssert() guarded by "
            "LEGATE_USE_DEBUG in the provided file(s)."
        ),
        replacements=repl,
    ).main()


if __name__ == "__main__":
    sys.exit(main())
