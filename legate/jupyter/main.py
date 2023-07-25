#!/usr/bin/env python

# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

from legate.driver import LegateDriver
from legate.jupyter.config import Config
from legate.jupyter.kernel import generate_kernel_spec, install_kernel_spec
from legate.util.system import System

__all__ = ("main",)


def main(argv: list[str]) -> int:
    config = Config(argv)
    system = System()

    driver = LegateDriver(config, system)

    spec = generate_kernel_spec(driver, config)

    install_kernel_spec(spec, config)

    return 0
