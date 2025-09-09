# SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from .context import Context


def bump_cmakelists_version(ctx: Context) -> None:
    # Bump the top-level VERSION file used by CMake.
    version_file: Path = ctx.legate_dir / "VERSION"
    ctx.vprint(f"Opening {version_file}")
    full_version = ctx.to_full_version(
        ctx.version_after_this, extra_zeros=True
    )
    if not ctx.dry_run:
        version_file.write_text(f"{full_version}\n")
    ctx.vprint(f"Updated {version_file}")


def bump_legion_version(ctx: Context) -> None:
    legion_version = (
        ctx.legate_dir / "src" / "cmake" / "versions" / "legion_version.json"
    )
    assert legion_version.is_file()
    ctx.vprint(f"Opening {legion_version}")
    with legion_version.open() as fd:
        data = json.load(fd)

    lg_data = data["packages"]["Legion"]
    full_ver = ctx.to_full_version(ctx.version_after_this)
    assert "version" in lg_data
    if lg_data["version"] == full_ver:
        ctx.vprint("Legion version already bumped")
        return

    lg_data["version"] = full_ver

    if not ctx.dry_run:
        with legion_version.open(mode="w") as fd:
            json.dump(data, fd, indent=4)
    ctx.vprint(f"Updated {legion_version}")
