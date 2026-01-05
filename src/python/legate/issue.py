# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

# ruff: noqa: D103, T201
import os

from .util.info import (
    print_build_info,
    print_package_details,
    print_package_versions,
    print_system_info,
)


def main() -> None:
    # legate-issue should never fail, but sometimes auto-config is
    # too aggressive and will cause legate-issue itself to crash
    os.environ["LEGATE_AUTO_CONFIG"] = "0"

    # we could just call _nested_pretty_print(info()), but
    # there is a noticeable delay in generating the package details
    # from calling conda, and we want to start printing information
    # as quickly as possible
    print_system_info()
    print()
    print_package_versions()
    print()
    print_build_info()
    print()
    print_package_details()
