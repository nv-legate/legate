#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys

from util.re_replacement import RegexReplacement, Replacement


def main() -> int:
    return RegexReplacement(
        description='Find "" includes and transform them to <> includes',
        replacements=[
            Replacement(
                r"#include\s+\"(.+)\"",
                r"#include <\1>",
                pragma_keyword="include",
            )
        ],
    ).main()


if __name__ == "__main__":
    sys.exit(main())
