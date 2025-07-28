#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version
from subprocess import check_call

from packaging.version import Version


def ensure_aedifix() -> None:
    # update pre-commit-config.yml as well in case this is changed
    VERSION = Version("1.9.0")

    try:
        mod_version = Version(version("aedifix"))

        if mod_version == VERSION:
            return

        if mod_version.is_devrelease:
            # If its a "dev release" that means it's editable installed,
            # meaning someone is working on aedifix. We don't care that the
            # versions don't match in this case.
            return

        raise RuntimeError  # noqa: TRY301
    except (PackageNotFoundError, RuntimeError):
        package = f"git+https://github.com/nv-legate/aedifix@{VERSION}"
        check_call([sys.executable, "-m", "pip", "install", package])
