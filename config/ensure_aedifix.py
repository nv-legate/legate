#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Final

# update pre-commit-config.yml as well in case this is changed
VERSION: Final = "1.0.0"


def ensure_aedifix() -> None:
    try:
        import aedifix

        if aedifix.__version__ != VERSION:
            raise RuntimeError  # noqa: TRY301

    except (ImportError, RuntimeError):
        import sys
        from subprocess import check_call

        package = f"git+https://github.com/nv-legate/aedifix@{VERSION}"
        check_call([sys.executable, "-m", "pip", "install", package])
