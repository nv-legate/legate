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

# Since this file sits top-level, a simple "import legate" would ALWAYS import
# pwd/legate/__init__.py. This behavior is not desirable/surprising if the user
# has already legate.core.
#
# To get around this, we move the current directory to the end of the module
# search path. That way the global modules are searched first, and the legate
# directory does not shadow the installed version.
del sys.path[0]
sys.path.append("")

# These are all still "top of file", but flake8 is feeling feisty about it
from legate.tester.args import parser  # noqa E402
from legate.tester.config import Config  # noqa E402
from legate.tester.test_plan import TestPlan  # noqa E402
from legate.tester.test_system import TestSystem  # noqa E402


def main() -> int:
    parser.set_defaults(
        gtest_file=GTEST_TESTS_BIN,
        mpi_output_filename=(
            GTESTS_TEST_DIR / "mpi_result" if GTESTS_TEST_DIR else None
        ),
    )

    config = Config(sys.argv)
    system = TestSystem(dry_run=config.dry_run)
    plan = TestPlan(config, system)

    return plan.execute()


def _find_latest_cpp_test_dir() -> tuple[Path, Path] | tuple[None, None]:
    if not (LEGATE_CORE_ARCH := os.environ.get("LEGATE_CORE_ARCH")):
        return None, None

    from scripts.get_legate_core_dir import get_legate_core_dir

    LEGATE_CORE_DIR = Path(get_legate_core_dir())

    lg_arch_dir = LEGATE_CORE_DIR / LEGATE_CORE_ARCH

    def make_test_dir(prefix: Path) -> Path:
        return prefix / "tests" / "cpp"

    def make_test_bin(prefix: Path) -> Path:
        return prefix / "bin" / "cpp_tests"

    def get_cpp_lib_dir() -> tuple[Path, Path] | None:
        cpp_lib = make_test_dir(lg_arch_dir / "cmake_build")
        cpp_bin = make_test_bin(cpp_lib)
        if cpp_bin.exists():
            return cpp_lib, cpp_bin
        return None

    def get_py_lib_dir() -> tuple[Path, Path] | None:
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
            skbuild_base / "cmake-build" / "legate-core-cpp"
        )
        py_bin = make_test_bin(py_lib)
        if py_bin.exists():
            return py_lib, py_bin
        return None

    if (cpp_exists := get_cpp_lib_dir()) is not None:
        cpp_lib_dir, cpp_bin = cpp_exists
    if (py_exists := get_py_lib_dir()) is not None:
        py_lib_dir, py_bin = py_exists

    if cpp_exists and py_exists:
        if cpp_bin.stat().st_mtime > py_bin.stat().st_mtime:
            return cpp_lib_dir, cpp_bin
        return py_lib_dir, py_bin
    elif cpp_exists:
        return cpp_lib_dir, cpp_bin
    elif py_exists:
        return py_lib_dir, py_bin
    return None, None


GTESTS_TEST_DIR, GTEST_TESTS_BIN = _find_latest_cpp_test_dir()

if __name__ == "__main__":
    sys.exit(main())
