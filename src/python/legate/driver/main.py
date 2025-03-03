# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from rich import print as rich_print

from . import LegateDriver

__all__ = ("legate_main",)


def prepare_driver(argv: list[str]) -> LegateDriver:
    from ..util.system import System
    from ..util.ui import error
    from . import Config
    from .driver import format_verbose

    try:
        config = Config(argv)
    except Exception:
        rich_print(error("Could not configure driver:\n"))
        raise

    try:
        system = System()
    except Exception:
        rich_print(error("Could not determine System settings: \n"))
        raise

    try:
        driver = LegateDriver(config, system)
    except Exception:
        msg = "Could not initialize driver, path config and exception follow:"
        rich_print(error(msg))
        rich_print(format_verbose(system), flush=True)
        raise

    return driver


def legate_main(argv: list[str]) -> int:
    """A main function for the Legate driver that can be used programmatically
    or by entry-points.

    Parameters
    ----------
        argv : list[str]
            Command-line arguments to start the Legate driver with

    Returns
    -------
        int, a process return code

    """
    driver = prepare_driver(argv)
    return driver.run()
