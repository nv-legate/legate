#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

import os
import sys
from pathlib import Path

from legate.tester.args import parser
from legate.tester.config import Config
from legate.tester.test_plan import TestPlan
from legate.tester.test_system import TestSystem


def main() -> int:
    parser.set_defaults(
        gtest_files=GTEST_TESTS_BIN,
        mpi_output_filename=(
            GTEST_TESTS_DIR / "mpi_result" if GTEST_TESTS_DIR else None
        ),
    )

    config = Config(sys.argv)
    system = TestSystem(dry_run=config.dry_run)
    plan = TestPlan(config, system)

    return plan.execute()


def _find_latest_cpp_test_dir() -> tuple[Path, list[Path]] | tuple[None, None]:
    if not (LEGATE_ARCH := os.environ.get("LEGATE_ARCH")):
        return None, None

    from scripts.get_legate_dir import get_legate_dir

    LEGATE_DIR = Path(get_legate_dir())

    lg_arch_dir = LEGATE_DIR / LEGATE_ARCH

    def make_test_dir(prefix: Path) -> Path:
        return prefix / "tests"

    def make_test_bin(prefix: Path) -> list[Path]:
        return [
            prefix / "bin" / "tests_with_runtime",
            prefix / "bin" / "tests_wo_runtime",
            prefix / "bin" / "tests_non_reentrant",
        ]

    def get_cpp_lib_dir() -> tuple[Path, list[Path]] | None:
        cpp_lib = make_test_dir(lg_arch_dir / "cmake_build" / "src" / "cpp")
        cpp_bin = make_test_bin(cpp_lib)
        if all(p.exists() for p in cpp_bin):
            return cpp_lib, cpp_bin
        return None

    def get_py_lib_dir() -> tuple[Path, list[Path]] | None:
        # Skbuild puts everything under a
        # <os-name>-<os-version>-<cpu-arch>-<py-version> directory inside the
        # skbuild directory. Since we are not interested in reverse engineering
        # the entire naming scheme, we just find the one which matches the
        # python version (which is the most likely to change).
        py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        skbuild_base = lg_arch_dir / "_skbuild"
        try:
            for os_dir in skbuild_base.iterdir():
                if os_dir.is_dir() and os_dir.name.endswith(py_version):
                    skbuild_base /= os_dir
                    break
            else:
                # Didn't find it, just bail
                return None
        except FileNotFoundError:
            # skbuild_base does not exist
            return None
        py_lib = make_test_dir(
            skbuild_base
            / "cmake-build"
            / "src"
            / "python"
            / "legate_cpp"
            / "src"
            / "cpp"
        )
        py_bin = make_test_bin(py_lib)
        if all(p.exists() for p in py_bin):
            return py_lib, py_bin
        return None

    if (cpp_exists := get_cpp_lib_dir()) is not None:
        cpp_lib_dir, cpp_bin = cpp_exists
    if (py_exists := get_py_lib_dir()) is not None:
        py_lib_dir, py_bin = py_exists

    if cpp_exists and py_exists:
        if all(
            cpp_bin_exe.stat().st_mtime > py_bin_exe.stat().st_mtime
            for cpp_bin_exe, py_bin_exe in zip(cpp_bin, py_bin)
        ):
            return cpp_lib_dir, cpp_bin
        return py_lib_dir, py_bin
    if cpp_exists:
        return cpp_lib_dir, cpp_bin
    if py_exists:
        return py_lib_dir, py_bin
    return None, None


GTEST_TESTS_DIR, GTEST_TESTS_BIN = _find_latest_cpp_test_dir()

if __name__ == "__main__":
    sys.exit(main())
