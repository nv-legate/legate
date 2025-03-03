# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Final

from ...cmake import CMAKE_VARIABLE, CMakeString
from ...util.argument_parser import ArgSpec, ConfigArgument
from ..package import Package

if TYPE_CHECKING:
    from ...manager import ConfigurationManager


class OpenMP(Package):
    With_OpenMP: Final = ConfigArgument(
        name="--with-openmp",
        spec=ArgSpec(
            dest="with_openmp", type=bool, help="Build with OpenMP support."
        ),
        enables_package=True,
        primary=True,
    )
    OpenMP_VERSION = CMAKE_VARIABLE("OpenMP_VERSION", CMakeString)
    OpenMP_CXX_FLAGS = CMAKE_VARIABLE("OpenMP_CXX_FLAGS", CMakeString)

    def __init__(self, manager: ConfigurationManager) -> None:
        super().__init__(manager=manager, name="OpenMP")

    def summarize(self) -> str:
        r"""Summarize configured OpenMP.

        Returns
        -------
        summary : str
            The summary of OpenMP
        """
        if not self.state.enabled():
            return ""

        lines = []
        if version := self.manager.get_cmake_variable(self.OpenMP_VERSION):
            lines.append(("Version", version))
        if flags := self.manager.get_cmake_variable(self.OpenMP_CXX_FLAGS):
            lines.append(("Flags", flags))
        return self.create_package_summary(lines)


def create_package(manager: ConfigurationManager) -> OpenMP:
    return OpenMP(manager)
