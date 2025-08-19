# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from typing import Final

from aedifix import Package
from aedifix.cmake import CMAKE_VARIABLE, CMakePath, CMakeString
from aedifix.util.argument_parser import ArgSpec, ConfigArgument


class UCX(Package):
    name = "UCX"

    With_UCX: Final = ConfigArgument(
        name="--with-ucx",
        spec=ArgSpec(
            dest="with_ucx", type=bool, help="Build with UCX support."
        ),
        enables_package=True,
        primary=True,
    )

    ucx_ROOT: Final = ConfigArgument(
        name="--with-ucx-dir",
        spec=ArgSpec(
            dest="ucx_dir",
            type=Path,
            help="Path to UCX installation directory.",
        ),
        cmake_var=CMAKE_VARIABLE("ucx_ROOT", CMakePath),
        enables_package=True,
    )

    ucc_ROOT: Final = ConfigArgument(
        name="--with-ucc-dir",
        spec=ArgSpec(
            dest="ucc_dir",
            type=Path,
            help="Path to UCC installation directory.",
        ),
        cmake_var=CMAKE_VARIABLE("ucc_ROOT", CMakePath),
        enables_package=True,
    )

    ucx_VERSION = CMAKE_VARIABLE("ucx_VERSION", CMakeString)
    ucc_VERSION = CMAKE_VARIABLE("ucc_VERSION", CMakeString)

    def configure(self) -> None:
        r"""Configure UCX."""
        super().configure()
        if not self.state.enabled():
            return

        self.set_flag_if_user_set(self.ucx_ROOT, self.cl_args.ucx_dir)
        self.set_flag_if_user_set(self.ucc_ROOT, self.cl_args.ucc_dir)

    def summarize(self) -> str:
        r"""Summarize UCX.

        Returns
        -------
        summary : str
            The summary of UCX and UCC.
        """
        if not self.state.enabled():
            return ""

        lines = []
        if ucx_version := self.manager.get_cmake_variable(self.ucx_VERSION):
            lines.append(("UCX Version", ucx_version))
        if ucx_root := self.manager.get_cmake_variable(self.ucx_ROOT):
            lines.append(("UCX Root directory", ucx_root))
        if ucc_version := self.manager.get_cmake_variable(self.ucc_VERSION):
            lines.append(("UCC Version", ucc_version))
        if ucc_root := self.manager.get_cmake_variable(self.ucc_ROOT):
            lines.append(("UCC Root directory", ucc_root))
        return self.create_package_summary(lines)
