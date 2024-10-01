#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.aedifix.main import basic_configure  # noqa: E402
from config.legate_internal.main_package import Legate  # noqa: E402


def main() -> int:
    argv = [
        f"--LEGATE_ARCH={Path(__file__).stem}",
        "--build-type=debug",
        "--with-cuda",
        "--with-cuda-dir=/usr/local/cuda",
        "--with-cudac=/usr/local/cuda/bin/nvcc",
    ] + sys.argv[1:]
    return basic_configure(tuple(argv), Legate)


if __name__ == "__main__":
    sys.exit(main())
