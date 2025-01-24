# SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES.
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

import json
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Final

    from .context import Context


class SwitcherData(TypedDict):
    name: str
    preferred: bool
    url: str
    version: str


def rotate_switcher(ctx: Context) -> None:
    switcher_json = ctx.legate_dir / "docs" / "legate" / "switcher.json"
    ctx.vprint(f"Opening {switcher_json}")
    assert switcher_json.is_file()
    with switcher_json.open() as fd:
        data: list[SwitcherData] = json.load(fd)

    for sub_data in data:
        if "preferred" in sub_data:
            last_release = sub_data
            break
    else:
        m = "Failed to find previous release in switcher data"
        raise ValueError(m)

    expected_keys = {"name", "preferred", "url", "version"}
    if last_release.keys() != expected_keys:
        diff = expected_keys - last_release.keys()
        m = (
            f"Unexpected keys in switcher dict. Expected {expected_keys}, "
            f"found the following extra/missing keys {diff}"
        )
        raise ValueError(m)

    # error: Key "preferred" of TypedDict "SwitcherData" cannot be deleted
    #
    # Yes it can
    if not (preferred := last_release.pop("preferred")):  # type: ignore[misc]
        m = (
            "Last release was not preferred. Expected 'preferred': true, "
            f"found {preferred!r}"
        )
        raise ValueError(m)

    last_version = last_release["version"]
    if last_version == ctx.release_version:
        # nothing to do
        ctx.vprint(f"Last release version {last_version} == current version")
        return

    new_release: SwitcherData = {
        "name": ctx.release_version,
        "preferred": True,
        "url": last_release["url"].replace(last_version, ctx.release_version),
        "version": ctx.release_version,
    }

    data.append(new_release)
    if not ctx.dry_run:
        with switcher_json.open(mode="w") as fd:
            json.dump(data, fd, indent=4, sort_keys=True)

    ctx.vprint(f"Updated {switcher_json} to {ctx.release_version}")


DEFAULT_CHANGELOG: Final = """\
Changes: Latest Development Version
===================================

..
   STYLE:
   * Capitalize sentences.
   * Use the imperative tense: Add, Improve, Change, etc.
   * Use a period (.) at the end of entries.
   * Be concise yet informative.
   * If possible, provide an executive summary of the new feature, but do not
     just repeat its doc string. However, if the feature requires changes from
     the user, then describe those changes in detail, and provide examples of
     the changes required.


.. rubric:: General

C++
---

.. rubric:: General

.. rubric:: Data

.. rubric:: Mapping

.. rubric:: Partitioning

.. rubric:: Tasks

.. rubric:: Types

.. rubric:: Runtime

.. rubric:: Utilities

.. rubric:: I/O


Python
------

.. rubric:: General

.. rubric:: Data

.. rubric:: Mapping

.. rubric:: Partitioning

.. rubric:: Tasks

.. rubric:: Types

.. rubric:: Runtime

.. rubric:: Utilities

.. rubric:: I/O
""".strip()


def update_changelog(ctx: Context) -> None:
    def rotate_log_files() -> Path:
        dev_log = (
            ctx.legate_dir
            / "docs"
            / "legate"
            / "source"
            / "changes"
            / "dev.rst"
        )
        old_log = dev_log.with_stem(ctx.release_version.replace(".", ""))

        ctx.vprint(f"Opening {dev_log}")
        assert dev_log.is_file()

        txt = dev_log.read_text()
        if txt == DEFAULT_CHANGELOG:
            ctx.vprint("changelog already rotated")
            return old_log

        lines = txt.splitlines()
        assert lines[0] == DEFAULT_CHANGELOG[: DEFAULT_CHANGELOG.find("\n")]
        lines[0] = f"Changes: {ctx.release_version}"
        lines[1] = "=" * len(lines[0])

        if not ctx.dry_run:
            old_log.write_text("\n".join(lines))
        ctx.vprint(f"Wrote old log to {old_log}")
        ctx.run_cmd(["git", "add", str(old_log)])

        if not ctx.dry_run:
            dev_log.write_text(DEFAULT_CHANGELOG)
        ctx.vprint(f"Updated {dev_log}")
        return old_log

    def update_links(old_log: Path) -> None:
        index = old_log.parent / "index.rst"
        assert index.is_file()
        ctx.vprint(f"Opening {index}")
        lines = index.read_text().splitlines()
        for idx, line in enumerate(lines):
            if "In Development <dev.rst>" in line:
                idx += 1  # noqa: PLW2901
                break
        else:
            m = f"Failed to find link to 'In Development' changelog in {index}"
            raise ValueError(m)

        cur_line = lines[idx]
        padding = " " * (len(cur_line) - len(cur_line.lstrip()))
        template = f"{padding}{ctx.release_version} <{old_log.name}>"
        if cur_line == template:
            ctx.vprint("Link already updated")
            return

        lines.insert(idx, template)
        if not ctx.dry_run:
            index.write_text("\n".join(lines))
        ctx.vprint("Updated link to current documentation")

    old_log = rotate_log_files()
    update_links(old_log)
