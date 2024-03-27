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


class NCCL(Package):
    With_NCCL: Final = ConfigArgument(
        name="--with-nccl",
        spec=ArgSpec(
            dest="with_nccl", type=bool, help="Build with NCCL support."
        ),
    )
    NCCL_DIR: Final = ConfigArgument(
        name="--with-nccl-dir",
        spec=ArgSpec(
            dest="nccl_dir",
            type=Path,
            help="Path to NCCL installation directory.",
        ),
        cmake_var=CMAKE_VARIABLE("NCCL_DIR", CMakePath),
    )

    def __init__(self, manager: ConfigurationManager) -> None:
        super().__init__(manager=manager, name="NCCL")

    def configure(self) -> None:
        r"""Configure NCCL."""
        super().configure()
        if not self.state.enabled():
            return

        self.set_flag_if_user_set(self.NCCL_DIR, self.cl_args.nccl_dir)

    def summarize(self) -> str:
        r"""Summarize NCCL.

        Returns
        -------
        summary : str
            A summary of configured NCCL.
        """
        lines = []
        if self.state.enabled():
            if nccl_dir := self.manager.get_cmake_variable(self.NCCL_DIR):
                lines.append(("Root dir", nccl_dir))
        return self.create_package_summary(lines)


def create_package(manager: ConfigurationManager) -> NCCL:
    return NCCL(manager)
