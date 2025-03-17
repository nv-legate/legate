#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
import argparse
from json import loads
from subprocess import check_output
from typing import TYPE_CHECKING, Any, Final, TypeAlias, cast

Issue: TypeAlias = dict[str, Any]

if TYPE_CHECKING:
    from collections.abc import Sequence

SRC_REPO: Final = "nv-legate/legate.internal"

PREFIX: Final = (
    "gh",
    "api",
    "-H",
    "Accept: application/vnd.github+json",
    "-H",
    "X-GitHub-Api-Version: 2022-11-28",
)


def execute(*cmd: str) -> Issue | list[Issue]:
    return loads(check_output(PREFIX + cmd, text=True))


def get_issue(issue_number: int) -> Issue:
    return cast(Issue, execute(f"/repos/{SRC_REPO}/issues/{issue_number}"))


def get_sub_issues(issue_number: int) -> tuple[Issue, ...]:
    return tuple(
        get_issue(int(sub_issue["number"]))
        for sub_issue in cast(
            list[Issue],
            execute(f"/repos/{SRC_REPO}/issues/{issue_number}/sub_issues"),
        )
    )


def add_assignees(
    issue_number: int, assignees: Sequence[str], target_repo: str
) -> None:
    cmd = [
        "--method",
        "POST",
        f"/repos/{target_repo}/issues/{issue_number}/assignees",
    ]
    for login in assignees:
        cmd += ["-f", f"assignees[]={login}"]
    execute(*cmd)


def add_sub_issue(epic: Issue, issue: Issue, target_repo: str) -> None:
    execute(
        "--method",
        "POST",
        f"/repos/{target_repo}/issues/{epic['number']}/sub_issues",
        "-F",
        f"sub_issue_id={issue['id']}",
    )


def port_text(
    text: str, prev_release: str, curr_release: str, next_release: str
) -> str:
    return (
        text.replace(curr_release, next_release)
        .replace(curr_release.replace(".", ""), next_release.replace(".", ""))
        .replace(prev_release, curr_release)
        .replace(prev_release.replace(".", ""), curr_release.replace(".", ""))
        .replace("- [X]", "- [ ]")
        .replace("- [x]", "- [ ]")
    )


def clone_issue(
    issue: Issue,
    prev_release: str,
    curr_release: str,
    next_release: str,
    target_repo: str,
) -> Issue:
    new_title = port_text(
        issue["title"], prev_release, curr_release, next_release
    )
    assignees = tuple(user["login"] for user in issue["assignees"])
    cmd = [
        "--method",
        "POST",
        f"/repos/{target_repo}/issues",
        "-f",
        f"title={new_title}",
    ]
    if (body := issue.get("body")) is not None:
        new_body = port_text(body, prev_release, curr_release, next_release)
        cmd += ["-f", f"body={new_body}"]
    new_issue = cast(Issue, execute(*cmd))
    add_assignees(int(new_issue["number"]), assignees, target_repo)
    return new_issue


def yy_dot_mm(val: str) -> str:
    if not re.match(r"^[0-9][0-9]\.[0-9][0-9]$", val):
        msg = "Release versions must be in the form YY.MM"
        raise argparse.ArgumentTypeError(msg)
    return val


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the release epic for the upcoming release, by "
        "cloning the epic from the previous one"
    )
    parser.add_argument(
        "--prev-release",
        required=True,
        type=yy_dot_mm,
        help="Last existing release, in the form YY.MM",
    )
    parser.add_argument(
        "--curr-release",
        required=True,
        type=yy_dot_mm,
        help="Release for which we are producing the epic, in the form YY.MM",
    )
    parser.add_argument(
        "--next-release",
        required=True,
        type=yy_dot_mm,
        help="Next projected release, in the form YY.MM",
    )
    parser.add_argument(
        "--epic-number",
        required=True,
        type=int,
        help="Issue number for the previous release epic",
    )
    parser.add_argument(
        "--target-repo",
        default="nv-legate/legate.internal",
        help="Repository on which to create the new epic",
    )
    args = parser.parse_args()

    epic = get_issue(args.epic_number)
    assert args.prev_release in epic["title"]
    cloned_epic = clone_issue(
        epic,
        args.prev_release,
        args.curr_release,
        args.next_release,
        args.target_repo,
    )

    for issue in get_sub_issues(args.epic_number):
        cloned_issue = clone_issue(
            issue,
            args.prev_release,
            args.curr_release,
            args.next_release,
            args.target_repo,
        )
        add_sub_issue(cloned_epic, cloned_issue, args.target_repo)


if __name__ == "__main__":
    main()
