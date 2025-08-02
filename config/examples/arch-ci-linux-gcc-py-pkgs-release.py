#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.ensure_aedifix import ensure_aedifix  # noqa: E402

ensure_aedifix()

from aedifix.main import basic_configure  # noqa: E402
from config.legate_internal.main_package import Legate  # noqa: E402


def main() -> int:
    arch = Path(__file__).stem
    cxx_only_flags = {"-pedantic"}
    cxx_flags = [
        "-Wall",
        "-Wextra",
        "-Werror",
        "-Walloca",
        "-Wdeprecated",
        "-Wimplicit-fallthrough",
        "-fdiagnostics-show-template-tree",
        "-Wignored-qualifiers",
        "-Wmissing-field-initializers",
        "-Wshadow",
        "-pedantic",
        "-fstack-protector-strong",
    ]
    cuda_flags = [
        f"--compiler-options={flag}"
        for flag in cxx_flags
        if flag not in cxx_only_flags
    ]
    argv = [
        # legate args
        f"--LEGATE_ARCH={arch}",
        "--build-type=release",
        "--cmake-generator=Ninja",
        # compilers and flags
        "--with-cc=gcc",
        "--with-cxx=g++",
        "--with-cudac=nvcc",
        "--CFLAGS=-O3",
        "--CXXFLAGS=-O3",
        "--CUDAFLAGS=-O3",
        # common options
        "--with-python",
        "--with-tests",
        # compiler flags
        "--legate-cxx-flags=" + " ".join(cxx_flags),
        "--legate-cuda-flags=" + " ".join(cuda_flags),
    ] + sys.argv[1:]
    return basic_configure(tuple(argv), Legate)


if __name__ == "__main__":
    sys.exit(main())
