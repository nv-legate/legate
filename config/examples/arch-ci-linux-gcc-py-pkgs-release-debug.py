#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.aedifix.main import basic_configure  # noqa: E402
from config.legate_core_internal.main_package import LegateCore  # noqa: E402


def main() -> int:
    arch = Path(__file__).stem
    cxx_only_flags = {"-pedantic"}
    cxx_flags = [
        "-Wall",
        "-Wextra",
        "-Werror",
        "-fstack-protector",
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
        # core args
        f"--LEGATE_CORE_ARCH={arch}",
        "--build-type=debug",
        "--cmake-generator=Ninja",
        # compilers and flags
        "--with-cc=gcc",
        "--with-cxx=g++",
        "--with-cudac=nvcc",
        "--CFLAGS=-O3 -g -g3",
        "--CXXFLAGS=-O3 -g -g3",
        "--CUDAFLAGS=-O3 -g -lineinfo -Xcompiler -O3 -Xcompiler -g3",
        # common options
        "--with-python",
        "--with-tests",
        # compiler flags
        "--legate-core-cxx-flags=" + " ".join(cxx_flags),
        "--legate-core-cuda-flags=" + " ".join(cuda_flags),
    ] + sys.argv[1:]
    return basic_configure(tuple(argv), LegateCore)


if __name__ == "__main__":
    sys.exit(main())
