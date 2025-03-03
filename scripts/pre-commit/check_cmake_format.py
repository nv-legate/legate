#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys
import json
import shutil
import subprocess
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from pathlib import Path
from typing import Any


def parse_args() -> Namespace:
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="Check that cmake-format.json is up to date",
    )

    cmake_genparsers = shutil.which("cmake-genparsers")

    parser.add_argument(
        "-i",
        "--input-file",
        metavar="<path>",
        required=True,
        type=Path,
        help="Path to input cmake-format.json to compare against",
    )
    parser.add_argument(
        "--cmake-genparsers",
        metavar="<path>",
        type=Path,
        default=cmake_genparsers,
        required=cmake_genparsers is None,
        help="path to cmake-genparsers binary",
    )
    parser.add_argument(
        "files", type=Path, nargs="+", help="cmake paths to parse"
    )
    return parser.parse_args()


def parse_file(
    cmake_genparsers: Path, path: Path
) -> dict[str, dict[str, dict[str, str]]]:
    ret = subprocess.run(
        [cmake_genparsers, "-f", "json", path], capture_output=True, check=True
    ).stdout.decode()
    ret = ret.strip().removesuffix(str(path))
    return json.loads(ret)


def handle_file(
    cmake_genparsers: Path, path: Path, format_data: dict[str, Any]
) -> bool:
    ret = parse_file(cmake_genparsers, path)
    updated = False
    for fn, spec in ret.items():
        try:
            cur = format_data[fn]
        except KeyError:
            format_data[fn] = spec
            updated = True
            print("Adding missing function:", fn)  # noqa: T201
            continue

        if cur != spec:
            print("Updating function:", fn)  # noqa: T201
            format_data[fn] = spec
            updated = True

    return updated


def main() -> int:
    args = parse_args()

    with args.input_file.open() as fd:
        data = json.load(fd)

    format_data = data["parse"]["additional_commands"]

    updated = False
    for path in args.files:
        if handle_file(
            args.cmake_genparsers, path.resolve(strict=True), format_data
        ):
            updated = True

    if updated:
        with args.input_file.open("w") as fd:
            json.dump(data, fd, indent=4)
    return int(updated)


if __name__ == "__main__":
    sys.exit(main())
