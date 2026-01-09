#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.ensure_aedifix import ensure_aedifix  # noqa: E402

ensure_aedifix()

from config.legate_internal.main_package import Legate  # noqa: E402

from aedifix.main import basic_configure

DEFAULT_BUILD_MODE = "release"


def resolve_build_type() -> str:
    build_mode = os.environ.get("LEGATE_BUILD_MODE", "").strip()
    return build_mode.lower() if build_mode else DEFAULT_BUILD_MODE


def main() -> int:
    build_type = resolve_build_type()
    argv = [
        # legate args
        f"--LEGATE_ARCH={Path(__file__).stem}",
        f"--build-type={build_type}",
        "--cmake-generator=Ninja",
        # common options
        "--with-cuda=0",
        "--with-python",
        "--with-docs",
    ] + sys.argv[1:]
    return basic_configure(tuple(argv), Legate)


if __name__ == "__main__":
    sys.exit(main())
