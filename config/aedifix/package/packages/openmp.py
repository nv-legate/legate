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

import ctypes.util
from typing import TYPE_CHECKING, Final

from ...util.argument_parser import ArgSpec, ConfigArgument
from ..package import EnableState, Package

if TYPE_CHECKING:
    from ...manager import ConfigurationManager


class OpenMP(Package):
    With_OpenMP: Final = ConfigArgument(
        name="--with-openmp",
        spec=ArgSpec(
            dest="with_openmp", type=bool, help="Build with OpenMP support."
        ),
    )

    def __init__(self, manager: ConfigurationManager) -> None:
        super().__init__(manager=manager, name="OpenMP")

    def find_package(self) -> None:
        r"""Determine whether OpenMP is enabled. Does a very crude search for
        several common names of lib OpenMP.
        """
        super().find_package()
        for libname in ("omp", "libomp", "libgomp"):
            try:
                lib = ctypes.util.find_library(libname)
            except FileNotFoundError:
                # https://github.com/python/cpython/issues/114257
                continue
            if lib:
                self.log(
                    f"Found possible OpenMP library: {lib}, enabling OpenMP"
                )
                self._state = EnableState(value=True, explicit=False)
                break

    def summarize(self) -> str:
        r"""Summarize configured OpenMP.

        Returns
        -------
        summary : str
            The summary of OpenMP
        """
        lines = []
        if self.state.enabled():
            lines.append(("Enabled", True))
        return self.create_package_summary(lines)


def create_package(manager: ConfigurationManager) -> OpenMP:
    return OpenMP(manager)
