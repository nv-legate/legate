# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Final

from aedifix.cmake import CMAKE_VARIABLE, CMakeString
from aedifix.package import Package
from aedifix.util.argument_parser import ArgSpec, ConfigArgument


class OpenMP(Package):
    name = "OpenMP"

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
