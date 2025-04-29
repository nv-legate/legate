#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.ensure_aedifix import ensure_aedifix  # noqa: E402

ensure_aedifix()

from aedifix.main import basic_configure  # noqa: E402
from config.legate_internal.main_package import Legate  # noqa: E402


def main() -> int:
    argv = [
        f"--LEGATE_ARCH={Path(__file__).stem}",
        "--with-python",
        "--with-cuda",
        "--with-ucx",
        "--with-openmp",
        "--build-type=debug",
        "--with-tests",
        "--with-docs",
    ] + sys.argv[1:]
    return basic_configure(tuple(argv), Legate)


if __name__ == "__main__":
    sys.exit(main())
