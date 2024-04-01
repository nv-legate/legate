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
from argparse import Action, ArgumentParser, Namespace
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final

from ...cmake import CMAKE_VARIABLE, CMakeExecutable, CMakeList, CMakePath
from ...util.argument_parser import ArgSpec, ConfigArgument
from ..package import EnableState, Package

if TYPE_CHECKING:
    from ...manager import ConfigurationManager


def _guess_cuda_compiler() -> str | None:
    for env_guess in ("CUDAC", "CMAKE_CUDA_COMPILER"):
        if guess := os.environ.get(env_guess):
            return guess
    for ccguess in ("nvcc", "cudac"):
        if guess := shutil.which(ccguess):
            return guess
    return None


class CudaArchAction(Action):
    @staticmethod
    def map_cuda_arch_names(in_arch: str) -> list[str]:
        arch_map = {
            "pascal": "60",
            "volta": "70",
            "turing": "75",
            "ampere": "80",
            "ada": "89",
            "hopper": "90",
            # TODO(jfaibussowit): blackwell?
        }
        arch = []
        for sub_arch in in_arch.split(","):
            # support Turing, TURING, and, if the user is feeling spicy, tUrInG
            sub_arch = sub_arch.strip().casefold()
            if not sub_arch:
                # in_arch = "something,,something_else"
                continue
            arch.append(arch_map.get(sub_arch, sub_arch))
        return arch

    def __call__(
        self,
        parser: ArgumentParser,
        namespace: Namespace,
        values: str | Sequence[str] | None,
        option_string: str | None = None,
    ) -> None:
        if isinstance(values, (list, tuple)):
            str_values = ",".join(values)
        elif isinstance(values, str):
            str_values = values
        elif values is None:
            str_values = getattr(namespace, self.dest)
        else:
            raise TypeError(type(values))

        cuda_arch = self.map_cuda_arch_names(str_values)
        setattr(namespace, self.dest, cuda_arch)


class CUDA(Package):
    With_CUDA: Final = ConfigArgument(
        name="--with-cuda",
        spec=ArgSpec(
            dest="with_cuda",
            type=bool,
            default=shutil.which("nvcc") is not None,
            help="Build with CUDA support.",
        ),
    )
    CUDAToolkit_ROOT: Final = ConfigArgument(
        name="--with-cuda-dir",
        spec=ArgSpec(
            dest="cuda_dir",
            type=Path,
            default=os.environ.get("CUDA_HOME"),
            required=False,
            help="Path to CUDA installation directory.",
        ),
        cmake_var=CMAKE_VARIABLE("CUDAToolkit_ROOT", CMakePath),
    )
    CMAKE_CUDA_COMPILER: Final = ConfigArgument(
        name="--with-cudac",
        spec=ArgSpec(
            dest="CUDAC",
            type=Path,
            default=_guess_cuda_compiler(),
            help="Specify CUDA compiler",
        ),
        cmake_var=CMAKE_VARIABLE("CMAKE_CUDA_COMPILER", CMakeExecutable),
    )
    CMAKE_CUDA_FLAGS: Final = ConfigArgument(
        name="--CUDAFLAGS",
        spec=ArgSpec(dest="CUDAFLAGS", nargs=1, help="CUDA compiler flags"),
        cmake_var=CMAKE_VARIABLE("CMAKE_CUDA_FLAGS", CMakeList),
    )
    CMAKE_CUDA_ARCHITECTURES: Final = ConfigArgument(
        name="--cuda-arch",
        spec=ArgSpec(
            dest="cuda_arch",
            required=False,
            default="all-major",
            action=CudaArchAction,
            help=(
                "Specify the target GPU architecture. Available choices are: "
                "'all-major', 'all', 'native', a comma-separated list of "
                "numbers: '60' or '70, 80', or comma-separated list of names "
                "'ampere' or 'hopper, blackwell'"
            ),
        ),
        cmake_var=CMAKE_VARIABLE("CMAKE_CUDA_ARCHITECTURES", CMakeList),
    )

    def __init__(self, manager: ConfigurationManager) -> None:
        r"""Construct a CUDA Package.

        Parameters
        ----------
        manager : ConfigurationManager
            The configuration manager to manage this package.
        """
        super().__init__(manager=manager, name="CUDA")

    def find_package(self) -> None:
        r"""Attempt to find CUDA. Checks whether any of the locator flags has
        a truthy value, and assumes this means the package is enabled.
        """
        super().find_package()
        if self.state.enabled():
            return  # no need to search any more

        cl_args = self.cl_args
        for v in (cl_args.CUDAC,):
            if v.value:
                self.log(
                    f"Enabling CUDA because {v.name} has truthy "
                    f'value "{v.value}" ({v})'
                )
                self._enabled = EnableState(value=True, explicit=v.cl_set)
                break

    def configure(self) -> None:
        r"""Configure CUDA."""
        super().configure()
        if not self.state.enabled():
            return

        self.set_flag_if_user_set(self.CMAKE_CUDA_COMPILER, self.cl_args.CUDAC)
        self._configure_language_flags(
            self.CMAKE_CUDA_FLAGS, self.cl_args.CUDAFLAGS
        )

        self.cuda_arch = self.cl_args.cuda_arch
        self.append_flags_if_set(self.CMAKE_CUDA_ARCHITECTURES, self.cuda_arch)
        self.set_flag_if_user_set(self.CUDAToolkit_ROOT, self.cl_args.cuda_dir)

    def summarize(self) -> str:
        r"""Summarize CUDA.

        Returns
        -------
        summary : str
            A summary of configured CUDA.
        """
        if not self.state.enabled():
            return ""

        arches = self.manager.get_cmake_variable(self.CMAKE_CUDA_ARCHITECTURES)
        ret = [("Architectures", " ".join(arches))]
        if cuda_dir := self.manager.get_cmake_variable(self.CUDAToolkit_ROOT):
            ret.append(("CUDA Dir", cuda_dir))
        try:
            cc = self.manager.read_cmake_variable(self.CMAKE_CUDA_COMPILER)
        except ValueError:
            cc = self.cl_args.CUDAC.value

        ret.append(("Executable", cc))
        ccflags: str | None | list[str] | tuple[str, ...]
        try:
            ccflags = self.manager.read_cmake_variable(self.CMAKE_CUDA_FLAGS)
        except ValueError:
            ccflags = self.cl_args.CUDAFLAGS.value
            if isinstance(ccflags, (list, tuple)):
                ccflags = " ".join(ccflags)

        if not ccflags:
            ccflags = "[]"
        ret.append(("Flags", ccflags))
        return self.create_package_summary(ret)


def create_package(manager: ConfigurationManager) -> CUDA:
    return CUDA(manager)
