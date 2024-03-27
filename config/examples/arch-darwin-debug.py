#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.aedifix.main import basic_configure  # noqa: E402
from config.legate_core_internal.main_package import LegateCore  # noqa: E402


def main() -> int:
    argv = [
        f"--LEGATE_CORE_ARCH={Path(__file__).stem}",
        "--with-cc=clang",
        "--with-cxx=clang++",
        "--build-type=debug",
        "--CFLAGS=-O0 -g3",
        "--CXXFLAGS=-O0 -g3",
        "--legate-core-cxx-flags=-Wall -Werror -fsanitize=address,undefined,bounds",  # noqa: E501
        "--legate-core-linker-flags=-fsanitize=address,undefined,bounds -fno-sanitize-recover=undefined",  # noqa: E501
        "--legion-bounds-check",
    ] + sys.argv[1:]
    return basic_configure(tuple(argv), LegateCore)


if __name__ == "__main__":
    sys.exit(main())
