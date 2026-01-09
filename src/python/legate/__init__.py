# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)


# Setuptools_scm will only format calver versions properly e.g. "25.03"
# instead of "25.3", with the "calver-by-date" version scheme. Unfortunately
# this mode also unconditionally always uses the current date as input, which
# is too inflexible for our use. So, we fix up the version format ourselves.
def _fixup_version() -> str:
    import os  # noqa: PLC0415

    if (v := os.environ.get("LEGATE_USE_VERSION")) is not None:
        return v

    try:
        from ._version import __version_tuple__ as vt  # noqa: PLC0415
    except ModuleNotFoundError:
        from datetime import datetime  # noqa: PLC0415

        # We haven't built the python bindings yet, so just construct the
        # version from current year and month. The actual version string
        # shouldn't matter because if we haven't built the python side, then
        # surely nobody (external) would be inspecting this anyways.
        rn = datetime.now()
        return f"{rn.strftime('%y')}.{rn.month:02}.0.dev"

    calver_base = ".".join(f"{x:02}" for x in vt[:3])
    dev = f".{vt[3]}" if len(vt) > 3 else ""  # noqa: PLR2004
    commit = f"+{vt[4]}" if len(vt) > 4 else ""  # noqa: PLR2004
    return calver_base + dev + commit


__version__ = _fixup_version()
del _fixup_version
