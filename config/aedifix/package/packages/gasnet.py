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

from ...cmake import CMAKE_VARIABLE, CMakePath, CMakeString
from ...util.argument_parser import ArgSpec, ConfigArgument
from ..package import EnableState, Package

if TYPE_CHECKING:
    from ...manager import ConfigurationManager


class GASNet(Package):
    With_GASNET: Final = ConfigArgument(
        name="--with-gasnet",
        spec=ArgSpec(
            dest="with_gasnet", type=bool, help="Build with GASNet support."
        ),
    )
    GASNet_ROOT_DIR: Final = ConfigArgument(
        name="--with-gasnet-dir",
        spec=ArgSpec(
            dest="gasnet_dir",
            type=Path,
            help="Path to GASNet installation directory.",
        ),
        cmake_var=CMAKE_VARIABLE("GASNet_ROOT_DIR", CMakePath),
    )
    GASNet_CONDUIT: Final = ConfigArgument(
        name="--gasnet-conduit",
        spec=ArgSpec(
            dest="gasnet_conduit",
            # TODO: To support UDP conduit, we would need to add a special case
            # on the legate launcher. See
            # https://github.com/nv-legate/legate.core/issues/294.
            choices=("ibv", "ucx", "aries", "mpi", "ofi"),
            help="Build with specified GASNet conduit.",
        ),
        cmake_var=CMAKE_VARIABLE("GASNet_CONDUIT", CMakeString),
    )
    GASNet_SYSTEM: Final = ConfigArgument(
        name="--gasnet-system",
        spec=ArgSpec(
            dest="gasnet_system",
            help="Specify a system-specific configuration to use for GASNet",
        ),
        cmake_var=CMAKE_VARIABLE("GASNet_SYSTEM", CMakeString),
    )

    def __init__(self, manager: ConfigurationManager) -> None:
        r"""Construct a GASNet Package.

        Parameters
        ----------
        manager : ConfigurationManager
            The configuration manager to manage this package.
        """
        super().__init__(manager=manager, name="GASNet")

    def find_package(self) -> None:
        r"""Attempt to find GASNet."""
        super().find_package()
        if self.state.enabled():
            return

        cl_args = self.cl_args
        for v in (
            cl_args.gasnet_conduit,
            cl_args.gasnet_system,
        ):
            if value := v.value:
                self.log(
                    f"Enabling GASNet due to {v} having truthy value {value}"
                )
                self._enabled = EnableState(value=True, explicit=v.cl_set)
                break

    def configure(self) -> None:
        r"""Configure GASNet."""
        super().configure()
        if not self.state.enabled():
            return

        self.set_flag_if_user_set(
            self.GASNet_ROOT_DIR, self.cl_args.gasnet_dir
        )
        self.set_flag_if_user_set(
            self.GASNet_CONDUIT, self.cl_args.gasnet_conduit
        )
        self.set_flag_if_user_set(
            self.GASNet_SYSTEM, self.cl_args.gasnet_system
        )

    def summarize(self) -> str:
        r"""Summarize GASNet.

        Returns
        -------
        summary : str
            A summary of configured GASNet.
        """
        lines = []
        if root_dir := self.manager.get_cmake_variable(self.GASNet_ROOT_DIR):
            lines.append(("Root directory", root_dir))
        if conduit := self.manager.get_cmake_variable(self.GASNet_CONDUIT):
            lines.append(("Conduit(s)", conduit))
        if system := self.manager.get_cmake_variable(self.GASNet_SYSTEM):
            lines.append(("System", system))
        return self.create_package_summary(lines)


def create_package(manager: ConfigurationManager) -> GASNet:
    return GASNet(manager)
