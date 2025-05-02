# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Final

from aedifix.cmake import CMAKE_VARIABLE, CMakePath
from aedifix.package import Package
from aedifix.util.argument_parser import ArgSpec, ConfigArgument


class HDF5(Package):
    name = "HDF5"

    With_HDF5: Final = ConfigArgument(
        name="--with-hdf5",
        spec=ArgSpec(
            dest="with_hdf5",
            type=bool,
            help="Build with HDF5 support.",
            default=bool(shutil.which("h5dump")),
        ),
        enables_package=True,
        primary=True,
    )
    HDF5_ROOT: Final = ConfigArgument(
        name="--with-hdf5-dir",
        spec=ArgSpec(
            dest="hdf5_dir",
            type=Path,
            help="Path to HDF5 installation directory.",
        ),
        cmake_var=CMAKE_VARIABLE("HDF5_ROOT", CMakePath),
        enables_package=True,
    )
    HDF5_DIR: Final = CMAKE_VARIABLE("HDF5_DIR", CMakePath)

    def configure(self) -> None:
        r"""Configure HDF5."""
        super().configure()
        if not self.state.enabled():
            return

        self.set_flag_if_user_set(self.HDF5_ROOT, self.cl_args.hdf5_dir)

    def summarize(self) -> str:
        r"""Summarize HDF5.

        Returns
        -------
        summary : str
            A summary of configured HDF5.
        """
        if not self.state.enabled():
            return ""

        lines = []

        def get_root_dir() -> str:
            root_dir = self.manager.get_cmake_variable(self.HDF5_ROOT)
            if not root_dir:
                root_dir = self.manager.get_cmake_variable(self.HDF5_DIR)
            return root_dir

        if root_dir := get_root_dir():
            lines.append(("Root directory", root_dir))
        return self.create_package_summary(lines)
