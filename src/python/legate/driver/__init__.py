# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from .config import Config
from .driver import LegateDriver
from .launcher import Launcher

__all__ = ("Config", "Launcher", "LegateDriver", "main")


def main() -> int:
    """Main function for legate entry point."""
    import os
    import sys
    import shlex

    from .main import legate_main as _main

    # A little explanation. We want to encourage configuration options be
    # passed via LEGATE_CONFIG, in order to be considerate to user scripts.
    # But we still need to accept actual command line args for compatibility,
    # and those should also take precedences. Here we splice the options from
    # LEGATE_CONFIG in before sys.argv, and take advantage of the fact that if
    # there are any options repeated in both places, argparse will use the
    # latter (i.e. the actual command line provided ones).
    env_args = shlex.split(os.environ.get("LEGATE_CONFIG", ""))
    argv = sys.argv[:1] + env_args + sys.argv[1:]

    return _main(argv)
