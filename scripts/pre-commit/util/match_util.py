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

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from pathlib import Path
    from re import Match


def is_in_comment(re_match: Match) -> bool:
    string = re_match.string
    prev_end = re_match.start()
    line_begin = string.rindex("\n", 0, prev_end) + 1
    line_prefix = string[line_begin:prev_end]
    return line_prefix.lstrip().startswith(("//", "/*", "*"))


HEADER_SUFFIXES: Final = {
    ".h",
    ".inl",
    ".cuh",
    ".cuinl",
    ".hpp",
    ".inc",
    ".HH",
}


def is_header(path: Path) -> bool:
    return path.suffix in HEADER_SUFFIXES


def is_publically_accessible(path: Path) -> bool:
    return "detail" not in path.parts


def is_in_test_dir(path: Path) -> bool:
    return "tests" in path.parts
