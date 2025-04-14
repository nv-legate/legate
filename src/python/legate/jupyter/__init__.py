# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from legate.jupyter.magic import LegateInfoMagics

if TYPE_CHECKING:
    from IPython import InteractiveShell

__all__ = ("load_ipython_extension", "main")


def load_ipython_extension(ipython: InteractiveShell) -> None:  # noqa: D103
    ipython.register_magics(LegateInfoMagics(ipython))


def main() -> int:  # noqa: D103
    import sys

    from .main import main as _main

    return _main(sys.argv)
