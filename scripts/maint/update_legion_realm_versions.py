# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import re
import json
import logging
import tempfile
import subprocess
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[2]
LEGION_VERSION_JSON = (
    REPO_ROOT / "src" / "cmake" / "versions" / "legion_version.json"
)
REALM_VERSION_JSON = (
    REPO_ROOT / "src" / "cmake" / "versions" / "realm_version.json"
)

GITLAB_PROJECT = "StanfordLegion/legion"
GITLAB_BASE = "https://gitlab.com/api/v4/projects"
REALM_GIT_URL = "https://github.com/StanfordLegion/realm.git"
HTTP_TIMEOUT = 15
LOGGER = logging.getLogger(__name__)


def fetch_legion_version(git_tag: str) -> str:
    encoded_project = quote(GITLAB_PROJECT, safe="")
    url = (
        f"{GITLAB_BASE}/{encoded_project}/repository/files/VERSION/raw"
        f"?ref={git_tag}"
    )
    request = Request(  # noqa: S310
        url, headers={"User-Agent": "legate-renovate"}
    )
    try:
        with urlopen(  # noqa: S310
            request, timeout=HTTP_TIMEOUT
        ) as response:
            version = response.read().decode().strip()
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        message = f"Failed to fetch Legion VERSION for {git_tag} from {url}"
        raise RuntimeError(message) from exc
    return version.removeprefix("legion-")


def _run_git(args: list[str], *, cwd: Path) -> str:
    result = subprocess.run(
        args, cwd=str(cwd), check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _parse_realm_tag(tag: str) -> str:
    tag = tag.strip()
    if tag.startswith("legion-"):
        tag = tag[len("legion-") :]
    elif tag.startswith("v"):
        tag = tag[1:]
    match = re.match(r"^([0-9]+(?:\.[0-9]+)*)", tag)
    if not match:
        message = f"Unexpected Realm tag format: {tag}"
        raise ValueError(message)
    return match.group(1)


def fetch_realm_version(git_tag: str) -> str:
    with tempfile.TemporaryDirectory(prefix="legate-realm-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        _run_git(["git", "init", "-q"], cwd=tmp_path)
        _run_git(
            ["git", "remote", "add", "origin", REALM_GIT_URL], cwd=tmp_path
        )
        _run_git(
            [
                "git",
                "fetch",
                "--filter=blob:none",
                "--tags",
                "origin",
                git_tag,
            ],
            cwd=tmp_path,
        )
        tag = ""
        for match in ("v[0-9]*", "legion-[0-9]*"):
            try:
                tag = _run_git(
                    [
                        "git",
                        "describe",
                        "--tags",
                        "--abbrev=0",
                        "--match",
                        match,
                        git_tag,
                    ],
                    cwd=tmp_path,
                )
                break
            except subprocess.CalledProcessError:
                continue
        if not tag:
            message = f"No version tag found for Realm commit {git_tag}"
            raise RuntimeError(message)
    return _parse_realm_tag(tag)


def _update_version(json_path: Path, package: str, new_version: str) -> bool:
    data = json.loads(json_path.read_text())
    pkg = data["packages"][package]
    current_version = pkg["version"]
    if current_version == new_version:
        return False
    pkg["version"] = new_version
    json_path.write_text(json.dumps(data, indent=4) + "\n")
    return True


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    legion_data = json.loads(LEGION_VERSION_JSON.read_text())
    legion = legion_data["packages"]["Legion"]
    legion_version = fetch_legion_version(legion["git_tag"])
    legion_updated = _update_version(
        LEGION_VERSION_JSON, "Legion", legion_version
    )

    realm_data = json.loads(REALM_VERSION_JSON.read_text())
    realm = realm_data["packages"]["Realm"]
    realm_version = fetch_realm_version(realm["git_tag"])
    realm_updated = _update_version(REALM_VERSION_JSON, "Realm", realm_version)

    if legion_updated:
        LOGGER.info("Updated Legion version to %s", legion_version)
    else:
        LOGGER.info("Legion version already %s", legion_version)
    if realm_updated:
        LOGGER.info("Updated Realm version to %s", realm_version)
    else:
        LOGGER.info("Realm version already %s", realm_version)


if __name__ == "__main__":
    main()
