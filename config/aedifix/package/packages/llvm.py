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

import shutil
from typing import TYPE_CHECKING, Final

from ...cmake import CMAKE_VARIABLE, CMakeExecutable
from ...util.argument_parser import ArgSpec, ConfigArgument
from ..package import EnableState, Package

if TYPE_CHECKING:
    from ...manager import ConfigurationManager


_llvm_config = shutil.which("llvm-config")


class LLVM(Package):
    With_LLVM: Final = ConfigArgument(
        name="--with-llvm",
        spec=ArgSpec(
            dest="with_llvm",
            type=bool,
            default=_llvm_config is not None,
            help="Build with LLVM support.",
        ),
    )
    LLVM_CONFIG_EXECUTABLE: Final = ConfigArgument(
        name="--llvm-config-executable",
        spec=ArgSpec(
            dest="llvm_config_executable",
            default=_llvm_config,
            help="Path to llvm-config executable to locate LLVM",
        ),
        cmake_var=CMAKE_VARIABLE("LLVM_CONFIG_EXECUTABLE", CMakeExecutable),
    )

    def __init__(self, manager: ConfigurationManager) -> None:
        r"""Construct a LLVM Package.

        Parameters
        ----------
        manager : ConfigurationManager
            The configuration manager to manage this package.
        """
        super().__init__(manager=manager, name="LLVM")

    def find_package(self) -> None:
        r"""Try to locate LLVM."""
        super().find_package()
        cl_args = self.cl_args
        for v in (cl_args.llvm_config_executable,):
            if value := v.value:
                self.log(
                    f"Enabling LLVM due to {v.name} having truthy "
                    f'value "{value}" ({v})'
                )
                self._enabled = EnableState(value=True, explicit=v.cl_set)
                break

    def configure(self) -> None:
        r"""Configure LLVM."""
        super().configure()
        if not self.state.enabled():
            return

        self.set_flag_if_user_set(
            self.LLVM_CONFIG_EXECUTABLE,
            self.cl_args.llvm_config_executable,
        )


def create_package(manager: ConfigurationManager) -> LLVM:
    return LLVM(manager)
