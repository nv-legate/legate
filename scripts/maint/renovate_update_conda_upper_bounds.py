#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import re
import sys
import argparse
from pathlib import Path

_MARKER_PREFIX = "renovate-conda-bound:"
_MARKER_RE = re.compile(r"renovate-conda-bound:\s*dep=(?P<dep>[a-z0-9_]+)")
_UPPER_BOUND_RE = re.compile(r"(<=?)([0-9][0-9A-Za-z._+-]*)")
_KEY_RE_TEMPLATE = r"^\s*{dep}_version:\s*$"
_LIST_ITEM_RE = re.compile(
    r"^(?P<indent>\s*)-\s*(?P<quote>[\"'])(?P<value>.*?)(?P=quote)(?P<suffix>\s*(?:#.*)?)$"
)
_STRICT_VERSION_RE = re.compile(r"^\d+(?:\.\d+)*$")


def _parse_numeric_segments(version: str, *, context: str) -> list[int]:
    if not _STRICT_VERSION_RE.match(version):
        msg = (
            f"Unsupported {context} {version!r}; expected digits and dots "
            "only."
        )
        raise ValueError(msg)
    return [int(part) for part in version.split(".")]


def _upper_bound_precision(op: str, upper: str) -> tuple[int, int]:
    segments = _parse_numeric_segments(upper, context="upper bound")
    if op != "<":
        return len(segments), 0

    last_non_zero = None
    for index in range(len(segments) - 1, -1, -1):
        if segments[index] != 0:
            last_non_zero = index
            break

    precision = 1 if last_non_zero is None else last_non_zero + 1

    trailing_zeros = len(segments) - precision
    return precision, trailing_zeros


def _replace_last_upper_bound(constraint: str, *, new_version: str) -> str:
    matches = list(_UPPER_BOUND_RE.finditer(constraint))
    if not matches:
        return constraint

    for match in matches:
        _parse_numeric_segments(match.group(2), context="upper bound")

    last = matches[-1]
    op = last.group(1)
    old_upper = last.group(2)
    precision, trailing_zeros = _upper_bound_precision(op, old_upper)

    segments = _parse_numeric_segments(new_version, context="new version")
    new_len = len(segments)
    extra_segments = segments[precision:] if new_len > precision else []
    if new_len < precision:
        segments.extend([0] * (precision - len(segments)))
    else:
        segments = segments[:precision]

    if op == "<":
        segments[-1] += 1
        if trailing_zeros:
            segments.extend([0] * trailing_zeros)
    elif op == "<=" and new_len > precision and any(extra_segments):
        # Round up to the next representable value at this precision.
        segments[-1] += 1

    new_upper = ".".join(str(value) for value in segments)
    if new_upper == old_upper:
        return constraint

    start, end = last.span(2)
    return f"{constraint[:start]}{new_upper}{constraint[end:]}"


def _strip_marker_lines(
    lines: list[str], *, dep: str
) -> tuple[list[str], bool]:
    changed = False
    kept: list[str] = []
    for line in lines:
        if _MARKER_PREFIX in line:
            match = _MARKER_RE.search(line)
            if match and match.group("dep") == dep:
                changed = True
                continue
        kept.append(line)
    return kept, changed


def _update_file(path: Path, *, dep: str, new_version: str) -> bool:
    original = path.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)

    lines, stripped = _strip_marker_lines(lines, dep=dep)

    key_re = re.compile(_KEY_RE_TEMPLATE.format(dep=re.escape(dep)))
    changed = stripped
    i = 0
    while i < len(lines):
        if not key_re.match(lines[i]):
            i += 1
            continue

        j = i + 1
        while j < len(lines) and (
            lines[j].lstrip().startswith("#") or lines[j].strip() == ""
        ):
            j += 1

        while j < len(lines):
            match = _LIST_ITEM_RE.match(lines[j].rstrip("\n"))
            if match is None:
                break

            indent = match.group("indent")
            quote = match.group("quote")
            value = match.group("value")
            suffix = match.group("suffix")

            updated_value = _replace_last_upper_bound(
                value, new_version=new_version
            )
            if updated_value != value:
                lines[j] = f"{indent}- {quote}{updated_value}{quote}{suffix}\n"
                changed = True

            j += 1

        i = j

    updated = "".join(lines)
    if not changed:
        return False

    path.write_text(updated, encoding="utf-8")
    return True


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Update conda_build_config.yaml upper bounds (Renovate helper)."
        )
    )
    parser.add_argument(
        "--dep", required=True, help="Dependency name (e.g. numpy)"
    )
    parser.add_argument(
        "--new-version",
        required=True,
        help="New upstream version (e.g. 2.5.1)",
    )
    args = parser.parse_args()

    try:
        root = _repo_root()
        # Keep upper bounds consistent across all conda subpackages.
        for path in sorted(root.glob("conda/**/conda_build_config.yaml")):
            _update_file(path, dep=args.dep, new_version=args.new_version)
    except ValueError as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
