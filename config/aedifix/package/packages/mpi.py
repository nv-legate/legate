# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Final

from ...cmake import CMAKE_VARIABLE, CMakeExecutable, CMakePath
from ...util.argument_parser import ArgSpec, ConfigArgument
from ..package import Package

if TYPE_CHECKING:
    from ...manager import ConfigurationManager


def _gen_mpiexec_guesses() -> str | None:
    for guess in ("mpiexec", "mpirun"):
        if found := shutil.which(guess):
            return found
    return None


class MPI(Package):
    With_MPI: Final = ConfigArgument(
        name="--with-mpi",
        spec=ArgSpec(
            dest="with_mpi", type=bool, help="Build with MPI support."
        ),
        enables_package=True,
        primary=True,
    )
    MPI_HOME: Final = ConfigArgument(
        name="--with-mpi-dir",
        spec=ArgSpec(
            dest="mpi_dir",
            type=Path,
            help="Path to MPI installation directory.",
        ),
        cmake_var=CMAKE_VARIABLE("MPI_HOME", CMakePath),
        enables_package=True,
    )
    MPIEXEC_EXECUTABLE: Final = ConfigArgument(
        name="--with-mpiexec-executable",
        spec=ArgSpec(
            dest="mpiexec",
            default=_gen_mpiexec_guesses(),
            type=Path,
            help="Path to mpiexec executable.",
        ),
        cmake_var=CMAKE_VARIABLE("MPIEXEC_EXECUTABLE", CMakeExecutable),
        enables_package=True,
    )
    MPI_CXX_COMPILER: Final = CMAKE_VARIABLE(
        "MPI_CXX_COMPILER", CMakeExecutable
    )
    MPI_CXX_COMPILER_INCLUDE_DIRS: Final = CMAKE_VARIABLE(
        "MPI_CXX_COMPILER_INCLUDE_DIRS", CMakePath
    )
    MPI_C_COMPILER: Final = CMAKE_VARIABLE("MPI_C_COMPILER", CMakeExecutable)
    MPI_C_COMPILER_INCLUDE_DIRS: Final = CMAKE_VARIABLE(
        "MPI_C_COMPILER_INCLUDE_DIRS", CMakePath
    )

    def __init__(self, manager: ConfigurationManager) -> None:
        r"""Construct a MPI Package.

        Parameters
        ----------
        manager : ConfigurationManager
            The configuration manager to manage this package.
        """
        super().__init__(manager=manager, name="MPI")

    def configure(self) -> None:
        r"""Configure MPI."""
        super().configure()
        if not self.state.enabled():
            return

        self.set_flag_if_user_set(self.MPI_HOME, self.cl_args.mpi_dir)
        self.set_flag_if_user_set(
            self.MPIEXEC_EXECUTABLE, self.cl_args.mpiexec
        )

    def summarize(self) -> str:
        r"""Summarize MPI.

        Returns
        -------
        summary : str
            A summary of configured MPI.
        """
        if not self.state.enabled():
            return ""

        lines = []
        if mpi_dir := self.manager.get_cmake_variable(self.MPI_HOME):
            lines.append(("Root dir", mpi_dir))
        if mpicc := self.manager.get_cmake_variable(self.MPI_C_COMPILER):
            lines.append(("mpicc", mpicc))
        if mpicc_inc := self.manager.get_cmake_variable(
            self.MPI_C_COMPILER_INCLUDE_DIRS
        ):
            lines.append(("C Include Dirs", mpicc_inc))
        if mpicxx := self.manager.get_cmake_variable(self.MPI_CXX_COMPILER):
            lines.append(("mpicxx", mpicxx))
        if mpicxx_inc := self.manager.get_cmake_variable(
            self.MPI_CXX_COMPILER_INCLUDE_DIRS
        ):
            lines.append(("C++ Include Dirs", mpicxx_inc))
        if mpiexec := self.manager.get_cmake_variable(self.MPIEXEC_EXECUTABLE):
            lines.append(("mpiexec", mpiexec))
        return self.create_package_summary(lines)


def create_package(manager: ConfigurationManager) -> MPI:
    return MPI(manager)
