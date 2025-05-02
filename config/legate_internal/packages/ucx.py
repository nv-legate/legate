# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from typing import Final

from aedifix.cmake import CMAKE_VARIABLE, CMakePath
from aedifix.package import Package
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
    UCX_ROOT: Final = ConfigArgument(
        name="--with-ucx-dir",
        spec=ArgSpec(
            dest="ucx_dir",
            type=Path,
            help="Path to UCX installation directory.",
        ),
        cmake_var=CMAKE_VARIABLE("UCX_ROOT", CMakePath),
        enables_package=True,
    )

    def configure(self) -> None:
        r"""Configure UCX."""
        super().configure()
        if self.state.enabled():
            return

        self.set_flag_if_user_set(self.UCX_ROOT, self.cl_args.ucx_dir)

    def summarize(self) -> str:
        r"""Summarize UCX.

        Returns
        -------
        summary : str
            The summary of UCX.
        """
        lines = []
        if self.state.enabled() and (
            root_dir := self.manager.get_cmake_variable(self.UCX_ROOT)
        ):
            lines.append(("Root dir", root_dir))
        return self.create_package_summary(lines)
