#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

try:
    # imported for effect
    import legate  # noqa: F401
except ModuleNotFoundError:
    from scripts.get_legate_dir import get_legate_dir

    sys.path.insert(0, str(Path(get_legate_dir()) / "src" / "python"))

from legate.tester.args import parser
from legate.tester.config import Config
from legate.tester.project import Project
from legate.tester.test_plan import TestPlan
from legate.tester.test_system import TestSystem


def _find_tests(prefix: Path) -> tuple[Path, list[Path]] | None:
    tests_dir = prefix / "cpp" / "tests"
    if not tests_dir.exists():
        return None

    ctest_file = tests_dir / "CTestTestfile.cmake"
    if not ctest_file.is_file():
        m = (
            f"Expected to find {ctest_file} in order to parse the tests from "
            "it, but this file does not exist. Perhaps the directory "
            "structure of the tests has changed?"
        )
        raise RuntimeError(m)

    tests_bin = [
        tests_dir / "bin" / p
        for p in re.findall(
            r"add_test\(\[=+\[(\w+)\]=+\]", ctest_file.read_text()
        )
    ]

    def should_remove(path: Path) -> bool:
        # rapids-cmake test spec generation
        return path.name == "generate_resource_spec"

    tests_bin = [p for p in tests_bin if not should_remove(p)]
    removed_bins = [p for p in tests_bin if not p.exists()]

    if not removed_bins:
        return tests_dir, tests_bin

    # some or all expected tests were removed
    m = (
        f"Missing expected test binary(s) {removed_bins}. They are listed "
        f"in {ctest_file} but don't appear to exist"
    )
    raise RuntimeError(m)


def _find_latest_cpp_test_dir() -> tuple[Path, list[Path]] | tuple[None, None]:
    if not (LEGATE_ARCH := os.environ.get("LEGATE_ARCH", "").strip()):
        try:
            from scripts.get_legate_arch import (  # type: ignore[import-not-found, unused-ignore]  # noqa: PLC0415
                get_legate_arch,
            )
        except ModuleNotFoundError:
            # User hasn't run configure yet, can't do anything
            return None, None

        LEGATE_ARCH = get_legate_arch()

    from scripts.get_legate_dir import get_legate_dir  # noqa: PLC0415

    legate_arch_dir = Path(get_legate_dir()) / LEGATE_ARCH

    cpp_ret = _find_tests(legate_arch_dir / "cmake_build")
    py_ret = _find_tests(
        legate_arch_dir / "skbuild_core" / "python" / "legate_cpp"
    )

    if cpp_ret and py_ret:
        cpp_build_is_newer = all(
            cpp_bin_exe.stat().st_mtime > py_bin_exe.stat().st_mtime
            for cpp_bin_exe, py_bin_exe in zip(
                cpp_ret[1], py_ret[1], strict=True
            )
        )
        if cpp_build_is_newer:
            return cpp_ret
        return py_ret

    if cpp_ret:
        return cpp_ret

    if py_ret:
        return py_ret

    return None, None


class LegateProject(Project):
    pass


def main() -> int:  # noqa: D103
    GTEST_TESTS_DIR, GTEST_TESTS_BIN = _find_latest_cpp_test_dir()

    parser.set_defaults(
        gtest_files=GTEST_TESTS_BIN,
        mpi_output_filename=(
            GTEST_TESTS_DIR / "mpi_result" if GTEST_TESTS_DIR else None
        ),
    )

    config = Config(sys.argv, project=LegateProject())
    system = TestSystem(dry_run=config.dry_run)
    plan = TestPlan(config, system)

    return plan.execute()


if __name__ == "__main__":
    sys.exit(main())
