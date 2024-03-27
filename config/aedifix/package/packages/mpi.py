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
from pathlib import Path
from typing import TYPE_CHECKING, Final

from ...cmake import CMAKE_VARIABLE, CMakeExecutable, CMakePath
from ...util.argument_parser import ArgSpec, ConfigArgument
from ..package import EnableState, Package

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
    )
    MPI_HOME: Final = ConfigArgument(
        name="--with-mpi-dir",
        spec=ArgSpec(
            dest="mpi_dir",
            type=Path,
            help="Path to MPI installation directory.",
        ),
        cmake_var=CMAKE_VARIABLE("MPI_HOME", CMakePath),
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
    )

    def __init__(self, manager: ConfigurationManager) -> None:
        r"""Construct a MPI Package.

        Parameters
        ----------
        manager : ConfigurationManager
            The configuration manager to manage this package.
        """
        super().__init__(manager=manager, name="MPI")

    def find_package(self) -> None:
        r"""Find MPI."""
        super().find_package()
        for v in (self.cl_args.mpiexec,):
            if value := v.value:
                self.log(
                    f"Enabling MPI due to {v.name} having "
                    f'truthy value "{value}" ({v})'
                )
                self._enabled = EnableState(value=True, explicit=v.cl_set)
                break

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
        lines = []
        if self.state.enabled():
            if mpi_dir := self.manager.get_cmake_variable(self.MPI_HOME):
                lines.append(("Root dir", mpi_dir))
            if mpiexec := self.manager.get_cmake_variable(
                self.MPIEXEC_EXECUTABLE
            ):
                lines.append(("mpiexec", mpiexec))
        return self.create_package_summary(lines)


def create_package(manager: ConfigurationManager) -> MPI:
    return MPI(manager)
