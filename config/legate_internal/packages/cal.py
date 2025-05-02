# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from typing import Final

from aedifix.cmake import CMAKE_VARIABLE, CMakePath
from aedifix.package import Package
from aedifix.util.argument_parser import ArgSpec, ConfigArgument


class CAL(Package):
    name = "CAL"

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

    def configure(self) -> None:
        r"""Configure CAL."""
        super().configure()
        if self.state.enabled():
            self.set_flag_if_user_set(self.CAL_DIR, self.cl_args.cal_dir)
