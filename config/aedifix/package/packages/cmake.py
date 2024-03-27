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

import os
import shutil
from typing import TYPE_CHECKING, Final

from ...cmake import CMAKE_VARIABLE, CMakePath, CMakeString
from ...util.argument_parser import ArgSpec, ConfigArgument
from ..package import Package

if TYPE_CHECKING:
    from ...manager import ConfigurationManager


_cmake_exe = shutil.which("cmake")


def _determine_default_generator() -> str | None:
    if ret := os.environ.get("CMAKE_GENERATOR"):
        return ret
    if shutil.which("ninja"):
        return "Ninja"
    if (
        shutil.which("make")
        or shutil.which("gmake")
        or shutil.which("gnumake")
    ):
        return "Unix Makefiles"
    return None


_default_gen = _determine_default_generator()


class CMake(Package):
    CMAKE_EXECUTABLE: Final = ConfigArgument(
        name="--cmake-executable",
        spec=ArgSpec(
            dest="cmake_executable",
            metavar="EXE",
            required=_cmake_exe is None,
            default=_cmake_exe,
            help="Path to CMake executable (if not on PATH).",
        ),
        cmake_var=CMAKE_VARIABLE("CMAKE_EXECUTABLE", CMakePath, prefix=""),
    )
    CMAKE_GENERATOR: Final = ConfigArgument(
        name="--cmake-generator",
        spec=ArgSpec(
            dest="cmake_generator",
            default=_default_gen,
            required=_default_gen is None,
            choices=["Ninja", "Unix Makefiles", None],
            help="The CMake build generator",
        ),
        cmake_var=CMAKE_VARIABLE("CMAKE_GENERATOR", CMakeString, prefix="-G"),
    )

    def __init__(self, manager: ConfigurationManager) -> None:
        r"""Construct a CMake Package.

        Parameters
        ----------
        manager : ConfigurationManager
            The configuration manager to manage this package.
        """
        super().__init__(manager=manager, name="CMake", always_enabled=True)

    def configure_cmake_version(self) -> None:
        r"""Determine the version of the cmake executable."""
        cmake_exe = self.manager.get_cmake_variable(self.CMAKE_EXECUTABLE)
        version = (
            self.log_execute_command([cmake_exe, "--version"])
            .stdout.splitlines()[0]  # "cmake version XX.YY.ZZ"
            .split()[2]  # ["cmake", "version", "XX.YY.ZZ"]
        )
        assert all(num.isdigit() for num in version.split("."))
        self.log(f"CMake executable version: {version}")
        self.version = version

    def configure_core_cmake_variables(self) -> None:
        r"""Configure the core cmake variables"""
        self.manager.set_cmake_variable(
            self.CMAKE_EXECUTABLE, self.cl_args.cmake_executable.value
        )
        self.manager.set_cmake_variable(
            self.CMAKE_GENERATOR, self.cl_args.cmake_generator.value
        )

    def register_gmake_subst(self) -> None:
        r"""Register gmake variable substitutions."""
        self.manager.add_gmake_search_variable(
            "CMAKE_COMMAND", project_var_name="CMAKE"
        )
        self.manager.add_gmake_search_variable("CMAKE_GENERATOR")

    def configure(self) -> None:
        r"""Configure CMake."""
        super().configure()
        self.log_execute_func(self.configure_core_cmake_variables)
        self.log_execute_func(self.configure_cmake_version)
        self.log_execute_func(self.register_gmake_subst)

    def summarize(self) -> str:
        r"""Summarize CMake.

        Returns
        -------
        summary : str
            A summary of configured CMake.
        """
        lines = [
            (
                "Executable",
                self.manager.get_cmake_variable(self.CMAKE_EXECUTABLE),
            ),
            ("Version", self.version),
            (
                "Generator",
                self.manager.get_cmake_variable(self.CMAKE_GENERATOR),
            ),
        ]

        return self.create_package_summary(lines)


def create_package(manager: ConfigurationManager) -> CMake:
    return CMake(manager)
