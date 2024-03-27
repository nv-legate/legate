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

import os
from pathlib import Path
from typing import TYPE_CHECKING, Final

from ...cmake import CMAKE_VARIABLE, CMakePath
from ...util.argument_parser import ArgSpec, ConfigArgument
from ..package import Package

if TYPE_CHECKING:
    from ...manager import ConfigurationManager


class Thrust(Package):
    With_Thrust: Final = ConfigArgument(
        name="--with-thrust",
        spec=ArgSpec(
            dest="with_thrust", type=bool, help="Build with Thrust support."
        ),
    )

    Thrust_ROOT: Final = ConfigArgument(
        name="--with-thrust-dir",
        spec=ArgSpec(
            dest="thrust_dir",
            type=Path,
            default=os.environ.get("THRUST_PATH"),
            help="Path to Thrust installation directory.",
        ),
        cmake_var=CMAKE_VARIABLE("Thrust_ROOT", CMakePath),
    )

    def __init__(self, manager: ConfigurationManager) -> None:
        super().__init__(manager=manager, name="Thrust")

    def configure(self) -> None:
        r"""Configure Thrust."""
        super().configure()
        self.set_flag_if_user_set(self.Thrust_ROOT, self.cl_args.thrust_dir)

    def summarize(self) -> str:
        r"""Summarize Thrust.

        Returns
        -------
        summary : str
            The summary of Thrust.
        """
        lines = []
        if root_dir := self.manager.get_cmake_variable(self.Thrust_ROOT):
            lines.append(("Root dir", root_dir))
        return self.create_package_summary(lines)


def create_package(manager: ConfigurationManager) -> Thrust:
    return Thrust(manager)
