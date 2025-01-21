#!/usr/bin/env python3
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

from argparse import ArgumentParser

from util.bump_cmake_versions import (
    bump_cmakelists_version,
    bump_legion_version,
)
from util.bump_docs_version import rotate_switcher, update_changelog
from util.context import Context


def parse_args() -> Context:
    parser = ArgumentParser()
    parser.add_argument(
        "--release-version",
        required=True,
        help="The release version to update to",
    )
    parser.add_argument(
        "--next-version", required=True, help="The next version"
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices={"pre-cut", "post-cut"},
        help="Release process mode",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    parser.add_argument("-n", "--dry-run", action="store_true", help="Dry-run")
    args = parser.parse_args()
    return Context(args)


def get_current_fork(ctx: Context) -> str:
    all_forks = ctx.run_cmd(
        [
            "gh",
            "repo",
            "list",
            "--fork",
            "--no-archived",
            "--json",
            "nameWithOwner",
            "-q",
            ".[].nameWithOwner",
        ]
    ).stdout

    names = ("legate.core.internal", "legate.internal")
    for repo in all_forks.splitlines():
        if any(n in repo for n in names):
            return repo

    m = f"Failed to find Legate fork in {all_forks}"
    raise ValueError(m)


def main() -> None:
    ctx = parse_args()

    match ctx.mode:
        case "pre-cut":
            rotate_switcher(ctx)
            update_changelog(ctx)
            bump_legion_version(ctx)
            bump_cmakelists_version(ctx)
        case "post-cut":
            cur_fork = get_current_fork(ctx)
            ctx.run_cmd(["gh", "repo", "sync", cur_fork])
            ctx.run_cmd(["git", "switch", "main"])
            ctx.run_cmd(["git", "pull"])
            ctx.run_cmd(["git", "pull", "upstream", "main"])
            ctx.run_cmd(
                [
                    "git",
                    "checkout",
                    "-b",
                    f"branch-{ctx.version}",
                    "upstream/main",
                ]
            )
            ctx.run_cmd(["git", "push", "upstream", "HEAD"])
        case _:
            raise ValueError(ctx.mode)


if __name__ == "__main__":
    main()
