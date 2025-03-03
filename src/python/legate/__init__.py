# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)


# Setuptools_scm will only format calver versions properly e.g. "25.03"
# instead of "25.3", with the "calver-by-date" version scheme. Unfortunately
# this mode also unconditionally always uses the current date as input, which
# is too inflexible for our use. So, we fix up the version format ourselves.
def _fixup_version() -> str:
    import os

    if (v := os.environ.get("LEGATE_USE_VERSION")) is not None:
        return v

    from ._version import __version_tuple__ as vt

    calver_base = ".".join(f"{x:02}" for x in vt[:3])
    dev = f".{vt[3]}" if len(vt) > 3 else ""  # noqa: PLR2004
    commit = f"+{vt[4]}" if len(vt) > 4 else ""  # noqa: PLR2004
    return calver_base + dev + commit


__version__ = _fixup_version()
del _fixup_version
