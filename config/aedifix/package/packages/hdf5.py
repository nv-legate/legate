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


class HDF5(Package):
    With_GASNET: Final = ConfigArgument(
        name="--with-hdf5",
        spec=ArgSpec(
            dest="with_hdf5", type=bool, help="Build with HDF5 support."
        ),
    )
    HDF5_ROOT: Final = ConfigArgument(
        name="--with-hdf5-dir",
        spec=ArgSpec(
            dest="hdf5_dir",
            type=Path,
            help="Path to HDF5 installation directory.",
        ),
        cmake_var=CMAKE_VARIABLE("HDF5_ROOT", CMakePath),
    )

    def __init__(self, manager: ConfigurationManager) -> None:
        r"""Construct a HDF5 Package.

        Parameters
        ----------
        manager : ConfigurationManager
            The configuration manager to manage this package.
        """
        super().__init__(manager=manager, name="HDF5")

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
        lines = []
        if self.state.enabled():
            if root_dir := self.manager.get_cmake_variable(self.HDF5_ROOT):
                lines.append(("Root directory", root_dir))
        return self.create_package_summary(lines)


def create_package(manager: ConfigurationManager) -> HDF5:
    return HDF5(manager)
