# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
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
    )
    CAL_DIR: Final = ConfigArgument(
        name="--with-cal-dir",
        spec=ArgSpec(
            dest="cal_dir",
            type=Path,
            help="Path to CAL installation directory.",
        ),
        cmake_var=CMAKE_VARIABLE("CAL_DIR", CMakePath),
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
