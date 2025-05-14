# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from typing import Final

from aedifix.cmake import CMAKE_VARIABLE, CMakePath
from aedifix.package import Package
from aedifix.packages.cuda import CUDA
from aedifix.util.argument_parser import ArgSpec, ConfigArgument


class NCCL(Package):
    name = "NCCL"

    dependencies = (CUDA,)

    With_NCCL: Final = ConfigArgument(
        name="--with-nccl",
        spec=ArgSpec(
            dest="with_nccl", type=bool, help="Build with NCCL support."
        ),
        enables_package=True,
        primary=True,
    )
    NCCL_DIR: Final = ConfigArgument(
        name="--with-nccl-dir",
        spec=ArgSpec(
            dest="nccl_dir",
            type=Path,
            help="Path to NCCL installation directory.",
        ),
        cmake_var=CMAKE_VARIABLE("NCCL_DIR", CMakePath),
        enables_package=True,
    )

    def configure(self) -> None:
        r"""Configure NCCL."""
        super().configure()
        # TODO(jfaibussowit)
        # Make this kind of relationship statically declarable from the CTOR,
        # by updating the dependencies argument to include a "this dependency
        # also enables the current package"
        if not self.state.explicit and self.deps.CUDA.state.enabled():
            self._state = Package.EnableState(value=True)
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
        if nccl_dir := self.manager.get_cmake_variable(self.NCCL_DIR):
            lines.append(("Root dir", nccl_dir))
        return self.create_package_summary(lines)
