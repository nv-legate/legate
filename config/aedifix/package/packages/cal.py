# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

from ...cmake import CMAKE_VARIABLE, CMakePath
from ...util.argument_parser import ArgSpec, ConfigArgument
from ..package import Package

if TYPE_CHECKING:
    from ...manager import ConfigurationManager


class CAL(Package):
    With_CAL: Final = ConfigArgument(
        name="--with-cal",
        spec=ArgSpec(
            dest="with_cal", type=bool, help="Build with CAL support."
        ),
        enables_package=True,
        primary=True,
    )
    CAL_DIR: Final = ConfigArgument(
        name="--with-cal-dir",
        spec=ArgSpec(
            dest="cal_dir",
            type=Path,
            help="Path to CAL installation directory.",
        ),
        cmake_var=CMAKE_VARIABLE("CAL_DIR", CMakePath),
        enables_package=True,
    )

    def __init__(self, manager: ConfigurationManager) -> None:
        r"""Construct a CAL Package.

        Parameters
        ----------
        manager : ConfigurationManager
            The configuration manager to manage this package.
        """
        super().__init__(manager=manager, name="CAL")

    def configure(self) -> None:
        r"""Configure CAL."""
        super().configure()
        if self.state.enabled():
            self.set_flag_if_user_set(self.CAL_DIR, self.cl_args.cal_dir)


def create_package(manager: ConfigurationManager) -> CAL:
    return CAL(manager)
