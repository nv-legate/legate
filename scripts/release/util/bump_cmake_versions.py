# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

import re
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .context import Context


def bump_cmakelists_version(ctx: Context, version: str) -> None:
    cmakelists = ctx.legate_dir / "src" / "CMakeLists.txt"
    ctx.vprint(f"Opening {cmakelists}")
    assert cmakelists.is_file()
    lines = cmakelists.read_text().splitlines()

    in_project = False
    for idx, line in enumerate(lines):
        if line.lstrip().startswith("project("):
            in_project = True
        if in_project:
            if "VERSION" in line:
                line_idx = idx
                break
            if ")" in line:
                in_project = False
    else:
        m = f"Failed to find project() call for legate in {cmakelists}"
        raise ValueError(m)

    full_version = ctx.to_full_version(version, extra_zeros=True)
    lines[line_idx] = re.sub(
        r"VERSION\s+\d+\.\d+\.\d+", f"VERSION {full_version}", lines[line_idx]
    )
    if not ctx.dry_run:
        cmakelists.write_text("\n".join(lines))
    ctx.vprint(f"Updated {cmakelists}")


def bump_legion_version(ctx: Context, version: str) -> None:
    legion_version = (
        ctx.legate_dir / "src" / "cmake" / "versions" / "legion_version.json"
    )
    assert legion_version.is_file()
    ctx.vprint(f"Opening {legion_version}")
    with legion_version.open() as fd:
        data = json.load(fd)

    lg_data = data["packages"]["Legion"]
    assert "version" in lg_data
    lg_data["version"] = ctx.to_full_version(version)

    if not ctx.dry_run:
        with legion_version.open(mode="w") as fd:
            json.dump(data, fd, indent=4)
    ctx.vprint(f"Updated {legion_version}")
