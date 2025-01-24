#!/usr/bin/env python3
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

from argparse import ArgumentParser

from packaging.version import Version, parse as parse_version

from util.bump_cmake_versions import (
    bump_cmakelists_version,
    bump_legion_version,
)
from util.bump_conda_versions import bump_legate_profiler_version
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
        choices={"pre-cut", "post-cut-release-branch", "post-cut-main-branch"},
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


def pre_cut(ctx: Context) -> None:
    rotate_switcher(ctx)
    update_changelog(ctx)
    bump_legion_version(ctx, ctx.release_version)
    bump_cmakelists_version(ctx, ctx.release_version)


def next_release_candidate(ctx: Context) -> int:
    all_tags = ctx.run_cmd(["git", "tag", "--list"]).stdout.splitlines()
    release_tags = (tag for tag in all_tags if ctx.release_version in tag)
    rc_tags = [tag for tag in release_tags if "rc" in tag.split(".")[-1]]
    if len(rc_tags) == 0:
        ctx.vprint(
            f"No prior releases for {ctx.release_version}, using release "
            "candidate 1"
        )
        return 1

    rc_tags.sort(key=Version)
    last_ver = parse_version(rc_tags[-1])
    ctx.vprint(f"Last release for {ctx.release_version}: {last_ver}")
    assert last_ver.pre is not None
    # last_ver.pre is e.g. ('rc', 3)
    _, old_rc = last_ver.pre
    assert isinstance(old_rc, int)
    new_rc = old_rc + 1
    ctx.vprint(f"New release for {ctx.release_version}: rc{new_rc}")
    return new_rc


def post_cut_release_branch(ctx: Context) -> None:
    cur_fork = get_current_fork(ctx)
    full_ver = ctx.to_full_version(ctx.release_version, extra_zeros=True)
    rc_ver = next_release_candidate(ctx)

    # Point of no return
    ctx.run_cmd(["gh", "repo", "sync", cur_fork])
    ctx.run_cmd(["git", "switch", "main"])
    ctx.run_cmd(["git", "pull"])
    ctx.run_cmd(["git", "pull", "upstream", "main"])
    ctx.run_cmd(
        [
            "git",
            "checkout",
            "-b",
            f"branch-{ctx.release_version}",
            "upstream/main",
        ]
    )
    ctx.run_cmd(["git", "push", "upstream", "HEAD"])
    ctx.run_cmd(["git", "tag", f"v{full_ver}.rc{rc_ver}", "HEAD"])
    ctx.run_cmd(["git", "push", "upstream", "--tags"])


def post_cut_main_branch(ctx: Context) -> None:
    bump_legate_profiler_version(ctx)
    bump_legion_version(ctx, ctx.next_version)
    bump_cmakelists_version(ctx, ctx.next_version)


def main() -> None:
    ctx = parse_args()

    match ctx.mode:
        case "pre-cut":
            pre_cut(ctx)
        case "post-cut-release-branch":
            post_cut_release_branch(ctx)
        case "post-cut-main-branch":
            post_cut_main_branch(ctx)
        case _:
            raise ValueError(ctx.mode)


if __name__ == "__main__":
    main()
