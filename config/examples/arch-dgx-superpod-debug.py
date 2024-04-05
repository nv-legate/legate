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
        # Specify the build type and enable extensive debugging
        "--build-type=debug",
        "--legion-bounds-check",
        # Enable GPUs
        "--with-cuda",
        "--cuda-arch=ampere",
        # Enable UCX
        "--with-ucx",
    ] + sys.argv[1:]
    return basic_configure(tuple(argv), LegateCore)


if __name__ == "__main__":
    sys.exit(main())
