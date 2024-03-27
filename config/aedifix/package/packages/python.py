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

import sys
from typing import TYPE_CHECKING, Final

from ...util.argument_parser import ArgSpec, ConfigArgument
from ...util.exception import UnsatisfiableConfigurationError
from ...util.utility import find_active_python_version_and_path
from ..package import Package

if TYPE_CHECKING:
    from ...manager import ConfigurationManager


class Python(Package):
    With_Python: Final = ConfigArgument(
        name="--with-python",
        spec=ArgSpec(
            dest="with_python", type=bool, help="Build with Python bindings."
        ),
    )

    def __init__(self, manager: ConfigurationManager) -> None:
        super().__init__(manager=manager, name="Python")

    def configure_lib_version_and_paths(self) -> None:
        r"""Determine the Python library version and its location."""
        try:
            version, lib_path = find_active_python_version_and_path()
        except (RuntimeError, FileNotFoundError) as excn:
            if self.state.disabled():
                # Not sure how we'd get here
                raise RuntimeError(
                    "The Python package does not appear to be enabled, yet we "
                    "are in the middle of configuring it. I'm not sure how we "
                    "got here, this should not happen"
                )
            # Python is requested, now to determine whether the user did
            # that or some other piece of the code
            if self.state.explicitly_enabled():
                # If the user wants python, but we cannot find/use it, then
                # that's a hard error
                raise UnsatisfiableConfigurationError(
                    f"{excn}. You have explicitly requested Python via "
                    f"--with-python {self.cl_args.with_python.value}"
                ) from excn
            # Some other piece of code has set the cl_args to true
            raise RuntimeError(
                f"{excn}. Some other package has implicitly enabled python"
                " but could not locate active lib directories for it"
            ) from excn

        self.lib_version = version
        self.lib_path = lib_path
        self.log(
            f"Python: found lib version: {version} and library path {lib_path}"
        )

    def configure(self) -> None:
        r"""Configure Python."""
        super().configure()
        if not self.state.enabled():
            return

        self.log_execute_func(self.configure_lib_version_and_paths)

    def summarize(self) -> str:
        r"""Summarize configured Python.

        Returns
        -------
        summary : str
            The summary of Python.
        """
        return self.create_package_summary(
            [("Executable", sys.executable), ("Version", sys.version)]
        )


def create_package(manager: ConfigurationManager) -> Python:
    return Python(manager)
