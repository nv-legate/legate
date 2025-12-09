# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from typing import Any

from gersemi.builtin_commands import builtin_commands


def make_rapids_find_package_() -> dict[str, Any]:
    orig = builtin_commands["find_package"]

    ret = deepcopy(orig)
    ret.setdefault("one_value_keywords", []).extend(
        ("GLOBAL_TARGETS", "BUILD_EXPORT_SET", "INSTALL_EXPORT_SET")
    )
    ret.setdefault("multi_value_keywords", []).extend(
        ("COMPONENTS", "FIND_ARGS")
    )
    ret.setdefault("sections", {})["FIND_ARGS"] = deepcopy(orig)
    return ret


command_definitions = {
    "rapids_cpm_find": {
        "front_positional_arguments": ["name", "version"],
        "options": ["BUILD_PATCH_ONLY"],
        "one_value_keywords": ["BUILD_EXPORT_SET", "INSTALL_EXPORT_SET"],
        "multi_value_keywords": ["COMPONENTS", "GLOBAL_TARGETS", "CPM_ARGS"],
        "sections": {
            "CPM_ARGS": {
                "one_value_keywords": [
                    "GIT_SHALLOW",
                    "GIT_REPOSITORY",
                    "SYSTEM",
                    "GIT_TAG",
                    "EXCLUDE_FROM_ALL",
                ],
                "multi_value_keywords": ["OPTIONS"],
            }
        },
    },
    "rapids_find_package": make_rapids_find_package_(),
}
