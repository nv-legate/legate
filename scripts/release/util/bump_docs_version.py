# SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from datetime import datetime
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

    dev_idx: int | None = None
    for idx, sub_data in enumerate(data):
        if sub_data.get("version") == "dev":
            dev_idx = idx
            break

    for sub_data in data:
        if sub_data.get("preferred", False):
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

    last_version = last_release["version"].strip()
    new_version = ctx.version_after_this
    if last_version == new_version:
        ctx.vprint(f"Switcher already points to {new_version}")
        return

    new_release: SwitcherData = {
        "name": new_version,
        "preferred": True,
        "url": last_release["url"].replace(last_version, new_version),
        "version": new_version,
    }

    data.append(new_release)
    if dev_idx is not None and 0 <= dev_idx < len(data) - 1:
        dev_entry = data.pop(dev_idx)
        data.append(dev_entry)
    if not ctx.dry_run:
        with switcher_json.open(mode="w") as fd:
            json.dump(data, fd, indent=4, sort_keys=True)

    ctx.vprint(f"Updated {switcher_json} to {new_version}")


DEFAULT_CHANGELOG: Final = """\
..
  SPDX-FileCopyrightText: Copyright (c) 2022-{current_year} NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

{header}
{underline}
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
""".strip()  # noqa: E501


def _rotate_log_files(ctx: Context) -> Path:
    ver_file = ctx.version_after_this.replace(".", "")
    new_log = (
        ctx.legate_dir
        / "docs"
        / "legate"
        / "source"
        / "changes"
        / f"{ver_file}.rst"
    )

    if new_log.is_file():
        ctx.vprint(f"Changelog already rotated ({new_log})")
        return new_log

    header = f"Changes: {ctx.version_after_this}"
    underline = "=" * len(header)
    current_year = datetime.now().year
    changelog = DEFAULT_CHANGELOG.format(
        header=header, underline=underline, current_year=current_year
    )

    if not ctx.dry_run:
        new_log.write_text(changelog)
        ctx.run_cmd(["git", "add", str(new_log)])

    ctx.vprint(f"Wrote new log to {new_log}")
    return new_log


def _update_symlink(ctx: Context, new_log: Path) -> None:
    dev_link = new_log.parent / "dev.rst"

    def make_link() -> None:
        if not ctx.dry_run:
            dev_link.symlink_to(new_log.name)
            ctx.run_cmd(["git", "add", str(dev_link)])
        ctx.vprint(f"Created symlink {dev_link} -> {new_log.name}")

    if not dev_link.exists():
        make_link()
        return

    assert dev_link.is_symlink()
    if dev_link.readlink().resolve() == new_log:
        ctx.vprint(f"dev link already exists {dev_link}")
        return

    dev_link.unlink()
    make_link()


def update_changelog(ctx: Context) -> None:
    new_log = _rotate_log_files(ctx)
    _update_symlink(ctx, new_log)
