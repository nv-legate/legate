# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from legate.driver import LegateDriver
from legate.jupyter.config import Config
from legate.jupyter.kernel import generate_kernel_spec, install_kernel_spec
from legate.util.system import System

__all__ = ("main",)


def main(argv: list[str]) -> int:  # noqa: D103
    config = Config(argv)
    system = System()

    driver = LegateDriver(config, system)

    spec = generate_kernel_spec(driver, config)

    install_kernel_spec(spec, config)

    return 0
