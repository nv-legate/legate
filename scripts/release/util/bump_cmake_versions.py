# SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import re
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Final

    from .context import Context


VERSION_RE: Final = re.compile(r"VERSION\s+(\d+\.\d+\.\d+)")


def get_cmakelists_version(
    src_file: Path, lines: list[str]
) -> tuple[str, int]:
    in_project = False
    for idx, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("project("):
            in_project = True

        if not in_project:
            continue

        if re_match := VERSION_RE.search(stripped):
            return (re_match[1], idx)

        if ")" in stripped:
            break

    m = f"Failed to find project() call for legate in {src_file}"
    raise ValueError(m)


def _do_bump_cmakelists_version(ctx: Context, cmakelists: Path) -> None:
    ctx.vprint(f"Opening {cmakelists}")
    lines = cmakelists.read_text().splitlines()
    _, idx = get_cmakelists_version(cmakelists, lines)
    full_version = ctx.to_full_version(
        ctx.version_after_this, extra_zeros=True
    )
    lines[idx] = re.sub(VERSION_RE, f"VERSION {full_version}", lines[idx])
    if not ctx.dry_run:
        cmakelists.write_text("\n".join(lines))
    ctx.vprint(f"Updated {cmakelists}")


def bump_cmakelists_version(ctx: Context) -> None:
    main_cmake_lists = ctx.legate_dir / "src" / "CMakeLists.txt"
    wheel_cmake_lists = (
        ctx.legate_dir
        / "scripts"
        / "build"
        / "python"
        / "legate"
        / "CMakeLists.txt"
    )
    for path in (main_cmake_lists, wheel_cmake_lists):
        _do_bump_cmakelists_version(ctx=ctx, cmakelists=path)


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
