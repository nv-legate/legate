#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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
