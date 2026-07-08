#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import re
import ast
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Final

CHECKED_CALLS: Final = {
    "pytest.mark.skip",
    "pytest.mark.skipif",
    "pytest.mark.xfail",
    "pytest.skip",
    "pytest.xfail",
}
PYTEST_MARK_CALLS: Final = {
    "skip": "pytest.mark.skip",
    "skipif": "pytest.mark.skipif",
    "xfail": "pytest.mark.xfail",
}
PYTEST_RUNTIME_CALLS: Final = {"skip": "pytest.skip", "xfail": "pytest.xfail"}
CATEGORY_RE: Final = re.compile(r"^\s*(?:not severe|severe):", re.IGNORECASE)
SEVERE_RE: Final = re.compile(r"^\s*severe:", re.IGNORECASE)
NOT_SEVERE_RE: Final = re.compile(r"^\s*not severe:", re.IGNORECASE)
ISSUE_RE: Final = re.compile(
    r"(?:https://github\.com/[^\s)]+/issues/\d+|#\d+|"
    r"issue[- ]?\d+|GH[- ]?\d+)",
    re.IGNORECASE,
)


class Options:
    def __init__(self, files: list[Path]) -> None:
        self.files = files


def parse_args() -> Options:
    parser = ArgumentParser(
        description=(
            "Require pytest skip/xfail markers to carry a reason, severity "
            "category, and GitHub issue for severe entries."
        )
    )
    parser.add_argument("files", nargs="*", type=Path)
    ns = parser.parse_args()
    return Options(ns.files)


def qualname(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = qualname(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def string_value(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(
                value.value, str
            ):
                parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                parts.append("{}")
            else:
                return None
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        lhs = string_value(node.left)
        rhs = string_value(node.right)
        if lhs is not None and rhs is not None:
            return lhs + rhs
    return None


def reason_for_call(call: ast.Call, name: str) -> str | None:
    for keyword in call.keywords:
        if keyword.arg == "reason":
            return string_value(keyword.value)

    # pytest.skip("reason") and pytest.xfail("reason") commonly use a
    # positional reason. pytest.mark.skip/xfail accept the same. skipif's first
    # positional argument is the condition, so only a second positional string
    # can be a reason there.
    reason_arg = 1 if name == "pytest.mark.skipif" else 0
    if len(call.args) > reason_arg:
        return string_value(call.args[reason_arg])
    return None


def pytest_aliases(tree: ast.AST) -> tuple[set[str], set[str], dict[str, str]]:
    pytest_names = {"pytest"}
    mark_names: set[str] = set()
    runtime_names: dict[str, str] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "pytest":
                    pytest_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "pytest":
            for alias in node.names:
                local_name = alias.asname or alias.name
                if alias.name == "mark":
                    mark_names.add(local_name)
                elif alias.name in PYTEST_RUNTIME_CALLS:
                    runtime_names[local_name] = PYTEST_RUNTIME_CALLS[
                        alias.name
                    ]

    return pytest_names, mark_names, runtime_names


def canonical_call_name(
    name: str,
    pytest_names: set[str],
    mark_names: set[str],
    runtime_names: dict[str, str],
) -> str:
    if name in runtime_names:
        return runtime_names[name]

    match name.split("."):
        case [mark_name, call_name] if mark_name in mark_names:
            return PYTEST_MARK_CALLS.get(call_name, name)
        case [pytest_name, call_name] if pytest_name in pytest_names:
            return PYTEST_RUNTIME_CALLS.get(call_name, name)
        case [pytest_name, "mark", call_name] if pytest_name in pytest_names:
            return PYTEST_MARK_CALLS.get(call_name, name)
        case _:
            return name


def iter_marker_calls(tree: ast.AST) -> list[tuple[ast.Call, str]]:
    calls: list[tuple[ast.Call, str]] = []
    pytest_names, mark_names, runtime_names = pytest_aliases(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = canonical_call_name(
                qualname(node.func), pytest_names, mark_names, runtime_names
            )
            if name in CHECKED_CALLS:
                calls.append((node, name))
    return calls


def validate_file(path: Path) -> list[str]:
    if not path.is_file() or path.suffix != ".py":
        return []

    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except SyntaxError as ex:
        return [f"{path}:{ex.lineno}: cannot parse Python: {ex.msg}"]

    errors: list[str] = []
    for call, name in iter_marker_calls(tree):
        reason = reason_for_call(call, name)
        location = f"{path}:{call.lineno}"

        if reason is None:
            errors.append(f"{location}: {name} must include a literal reason")
            continue

        if not reason.strip():
            errors.append(f"{location}: {name} reason must not be empty")
            continue

        has_not_severe = bool(NOT_SEVERE_RE.search(reason))
        has_severe = bool(SEVERE_RE.search(reason)) and not has_not_severe
        if not CATEGORY_RE.search(reason):
            errors.append(
                f"{location}: {name} reason must start with 'severe:' or "
                f"'not severe:'; got {reason!r}"
            )
            continue

        if has_severe and not ISSUE_RE.search(reason):
            errors.append(
                f"{location}: severe {name} reason must include a GitHub "
                f"issue reference; got {reason!r}"
            )
    return errors


def main() -> int:
    opts = parse_args()
    errors: list[str] = []
    for path in opts.files:
        errors.extend(validate_file(path))

    if errors:
        sys.stderr.write("pytest skip/xfail marker validation failed:\n")
        for error in errors:
            sys.stderr.write(f"  {error}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
