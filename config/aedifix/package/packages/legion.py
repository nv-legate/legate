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
from typing import TYPE_CHECKING, Any, Final, cast as TYPE_CAST

from ...cmake import (
    CMAKE_VARIABLE,
    CMakeBool,
    CMakeInt,
    CMakeList,
    CMakePath,
    CMakeString,
)
from ...util.argument_parser import (
    ArgSpec,
    ConfigArgument,
    ExclusiveArgumentGroup,
)
from ..package import Package

if TYPE_CHECKING:
    from ...manager import ConfigurationManager
    from .cuda import CUDA
    from .gasnet import GASNet
    from .hdf5 import HDF5
    from .llvm import LLVM
    from .mpi import MPI
    from .openmp import OpenMP
    from .python import Python
    from .ucx import UCX
    from .zlib import ZLIB


class Legion(Package):
    DirGroup: Final = ExclusiveArgumentGroup(
        Legion_ROOT=ConfigArgument(
            name="--with-legion-dir",
            spec=ArgSpec(
                dest="legion_dir",
                type=Path,
                help="Path to an existing Legion build directory.",
            ),
            cmake_var=CMAKE_VARIABLE("Legion_ROOT", CMakePath),
        ),
        CPM_Legion_SOURCE=ConfigArgument(
            name="--with-legion-src-dir",
            spec=ArgSpec(
                dest="legion_src_dir",
                type=Path,
                help="Path to an existing Legion source directory.",
            ),
            cmake_var=CMAKE_VARIABLE("CPM_Legion_SOURCE", CMakePath),
        ),
    )
    CPM_DOWNLOAD_Legion: Final = ConfigArgument(
        name="--legion-url",
        spec=ArgSpec(dest="legion_url", help="Legion git URL to build with."),
        cmake_var=CMAKE_VARIABLE("CPM_DOWNLOAD_Legion", CMakeBool),
    )
    Legion_BRANCH: Final = ConfigArgument(
        name="--legion-branch",
        spec=ArgSpec(
            dest="legion_branch", help="Git branch to download for Legion"
        ),
    )
    Legion_MAX_DIM: Final = ConfigArgument(
        name="--legion-max-dim",
        spec=ArgSpec(
            dest="legion_max_dim",
            type=int,
            default=4,
            help="Maximum number of dimensions that Legion will support",
        ),
        cmake_var=CMAKE_VARIABLE("Legion_MAX_DIM", CMakeInt),
    )
    Legion_MAX_FIELDS: Final = ConfigArgument(
        name="--legion-max-fields",
        spec=ArgSpec(
            dest="legion_max_fields",
            type=int,
            default=256,
            help="Maximum number of fields that Legion will support",
        ),
        cmake_var=CMAKE_VARIABLE("Legion_MAX_FIELDS", CMakeInt),
    )
    Legion_SPY: Final = ConfigArgument(
        name="--legion-spy",
        spec=ArgSpec(
            dest="legion_spy",
            type=bool,
            help="Build with detailed Legion Spy enabled.",
        ),
        cmake_var=CMAKE_VARIABLE("Legion_SPY", CMakeBool),
    )
    Legion_BOUNDS_CHECKS: Final = ConfigArgument(
        name="--legion-bounds-checks",
        spec=ArgSpec(
            dest="legion_bounds_checks",
            type=bool,
            help=(
                "Build Legion with bounds checking enabled "
                "(warning: expensive)."
            ),
        ),
        cmake_var=CMAKE_VARIABLE("Legion_BOUNDS_CHECKS", CMakeBool),
    )
    Legion_BUILD_RUST_PROFILER: Final = ConfigArgument(
        name="--legion-rust-profiler",
        spec=ArgSpec(
            dest="legion_rust_profiler",
            type=bool,
            help="Build the Legion profiler (requires rust).",
        ),
        cmake_var=CMAKE_VARIABLE("Legion_BUILD_RUST_PROFILER", CMakeBool),
    )
    Legion_CXX_FLAGS: Final = ConfigArgument(
        name="--legion-cxx-flags",
        spec=ArgSpec(
            dest="legion_cxx_flags", nargs=1, help="C++ flags for Legion"
        ),
        cmake_var=CMAKE_VARIABLE("Legion_CXX_FLAGS", CMakeList),
    )
    Legion_CUDA_FLAGS: Final = ConfigArgument(
        name="--legion-cuda-flags",
        spec=ArgSpec(
            dest="legion_cuda_flags", nargs=1, help="CUDA flags for Legion"
        ),
        cmake_var=CMAKE_VARIABLE("Legion_CUDA_FLAGS", CMakeList),
    )

    Legion_CUDA_ARCH: Final = CMAKE_VARIABLE("Legion_CUDA_ARCH", CMakeString)
    Legion_EMBED_GASNet_CONFIGURE_ARGS: Final = CMAKE_VARIABLE(
        "Legion_EMBED_GASNet_CONFIGURE_ARGS", CMakeList
    )
    Legion_USE_CUDA: Final = CMAKE_VARIABLE("Legion_USE_CUDA", CMakeBool)
    Legion_USE_OpenMP: Final = CMAKE_VARIABLE("Legion_USE_OpenMP", CMakeBool)
    Legion_USE_HDF5: Final = CMAKE_VARIABLE("Legion_USE_HDF5", CMakeBool)
    Legion_USE_LLVM: Final = CMAKE_VARIABLE("Legion_USE_LLVM", CMakeBool)
    Legion_USE_Python: Final = CMAKE_VARIABLE("Legion_USE_Python", CMakeBool)
    Legion_USE_ZLIB: Final = CMAKE_VARIABLE("Legion_USE_ZLIB", CMakeBool)
    Legion_Python_Version: Final = CMAKE_VARIABLE(
        "Legion_Python_Version", CMakeString
    )
    Legion_NETWORKS: Final = CMAKE_VARIABLE("Legion_NETWORKS", CMakeString)
    Legion_BUILD_JUPYTER: Final = CMAKE_VARIABLE(
        "Legion_BUILD_JUPYTER", CMakeBool
    )
    Legion_BUILD_BINDINGS: Final = CMAKE_VARIABLE(
        "Legion_BUILD_BINDINGS", CMakeBool
    )

    def __init__(self, manager: ConfigurationManager) -> None:
        r"""Construct a Legion Package.

        Parameters
        ----------
        manager : ConfigurationManager
            The configuration manager to manage this package.
        """
        super().__init__(manager=manager, name="Legion")

    def declare_dependencies(self) -> None:
        r"""Declare dependencies for Legion."""
        super().declare_dependencies()
        # mypy complains:
        #
        # config/aedifix/packages/legion.py: note: In member
        # "declare_dependencies" of class "Legion":
        # config/aedifix/packages/legion.py:127:27: error: Incompatible
        # types in assignment (expression has type "Package", variable has type
        # "CUDA")
        # [assignment]
        #    self.cuda: CUDA = self.require("cuda")
        #                      ^~~~~~~~~~~~~~~~~~~~
        #
        # So instead we first cast to Any, and then let mypy see the real type
        # on the lhs. We cannot just cast directly to the real type since that
        # means we'd actually have to import the class, which is a fake
        # dependency we don't want to introduce.
        self.cuda: CUDA = TYPE_CAST(Any, self.require("cuda"))
        self.gasnet: GASNet = TYPE_CAST(Any, self.require("gasnet"))
        self.openmp: OpenMP = TYPE_CAST(Any, self.require("openmp"))
        self.hdf5: HDF5 = TYPE_CAST(Any, self.require("hdf5"))
        self.llvm: LLVM = TYPE_CAST(Any, self.require("llvm"))
        self.python: Python = TYPE_CAST(Any, self.require("python"))
        self.mpi: MPI = TYPE_CAST(Any, self.require("mpi"))
        self.ucx: UCX = TYPE_CAST(Any, self.require("ucx"))
        self.zlib: ZLIB = TYPE_CAST(Any, self.require("zlib"))

    def configure_root_dirs(self) -> None:
        r"""Configure the various "root" directories that Legion requires"""
        dir_group = self.DirGroup
        if (lg_dir := self.cl_args.legion_dir).cl_set:
            self.manager.set_cmake_variable(
                dir_group.Legion_ROOT,  # type: ignore[attr-defined]
                lg_dir.value,
            )
        elif (lg_src_dir := self.cl_args.legion_src_dir).cl_set:
            self.manager.set_cmake_variable(
                dir_group.CPM_Legion_SOURCE,  # type: ignore[attr-defined]
                lg_src_dir.value,
            )

    def configure_variables(self) -> None:
        r"""Configure the variable default variables."""
        self.set_flag_if_user_set(
            self.Legion_MAX_DIM, self.cl_args.legion_max_dim
        )
        self.set_flag_if_user_set(
            self.Legion_MAX_FIELDS, self.cl_args.legion_max_fields
        )
        self.set_flag_if_user_set(self.Legion_SPY, self.cl_args.legion_spy)
        self.set_flag_if_user_set(
            self.Legion_BOUNDS_CHECKS, self.cl_args.legion_bounds_checks
        )
        self.set_flag_if_user_set(
            self.Legion_BUILD_RUST_PROFILER, self.cl_args.legion_rust_profiler
        )

        self.append_flags_if_set(
            self.Legion_CXX_FLAGS, self.cl_args.legion_cxx_flags
        )

    def configure_cuda(self) -> None:
        r"""If CUDA is enabled, set the various CUDA flags for Legion.
        Does nothing otherwise.

        Raises
        ------
        RuntimeError
            If CUDA flags are requested but CUDA is not enabled.
        """
        if self.cuda.state.enabled():
            self.manager.set_cmake_variable(self.Legion_USE_CUDA, True)
            self.append_flags_if_set(
                self.Legion_CUDA_FLAGS, self.cl_args.legion_cuda_flags
            )
            self.set_flag_if_user_set(
                self.Legion_CUDA_ARCH, self.cuda.cuda_arch
            )
        elif self.cl_args.legion_cuda_flags.cl_set:
            raise RuntimeError(
                "--legate-cuda-flags set "
                f"({self.cl_args.legion_cuda_flags.value}), "
                "but CUDA is not enabled."
            )
        elif self.cuda.state.explicitly_disabled():
            self.manager.set_cmake_variable(self.Legion_USE_CUDA, False)

    def configure_gasnet(self) -> None:
        r"""Configure Legion to use GASNet. Does nothing if GASNet is not
        enabled."""
        if self.gasnet.state.enabled():
            self.manager.append_cmake_variable(
                self.Legion_EMBED_GASNet_CONFIGURE_ARGS,
                ["--with-ibv-max-hcas=8"],
            )

    def configure_openmp(self) -> None:
        r"""Configure Legion to use OpenMP. Does nothing if OpenMP is not
        enabled."""
        if self.openmp.state.enabled():
            self.manager.set_cmake_variable(self.Legion_USE_OpenMP, True)
        elif self.openmp.state.explicitly_disabled():
            self.manager.set_cmake_variable(self.Legion_USE_OpenMP, False)

    def configure_hdf5(self) -> None:
        r"""Configure Legion to use HDF5. Does nothing if HDF5 is not
        enabled.
        """
        if self.hdf5.state.enabled():
            self.manager.set_cmake_variable(self.Legion_USE_HDF5, True)
        elif self.hdf5.state.explicitly_disabled():
            self.manager.set_cmake_variable(self.Legion_USE_HDF5, False)

    def configure_llvm(self) -> None:
        r"""Configure Legion to use LLVM. Does nothing if LLVM is not
        enabled.
        """
        if self.llvm.state.enabled():
            self.manager.set_cmake_variable(self.Legion_USE_LLVM, True)
        elif self.llvm.state.explicitly_disabled():
            self.manager.set_cmake_variable(self.Legion_USE_LLVM, False)

    def configure_python(self) -> None:
        r"""Configure Legion to use Python. Does nothing if Python is not
        enabled.
        """
        if self.python.state.enabled():
            self.manager.set_cmake_variable(self.Legion_BUILD_BINDINGS, True)
            self.manager.set_cmake_variable(self.Legion_USE_Python, True)
            self.manager.set_cmake_variable(self.Legion_BUILD_JUPYTER, True)
            self.manager.set_cmake_variable(
                self.Legion_Python_Version, self.python.lib_version
            )
        elif self.python.state.explicitly_disabled():
            self.manager.set_cmake_variable(self.Legion_BUILD_BINDINGS, False)
            self.manager.set_cmake_variable(self.Legion_BUILD_JUPYTER, False)
            self.manager.set_cmake_variable(self.Legion_USE_Python, False)

    def configure_zlib(self) -> None:
        r"""Configure Legion to use ZLIB. Disables ZLIB if is not enabled."""
        if self.zlib.state.enabled():
            self.manager.set_cmake_variable(self.Legion_USE_ZLIB, True)
        elif self.zlib.state.explicitly_disabled():
            self.manager.set_cmake_variable(self.Legion_USE_ZLIB, False)

    def configure_networks(self) -> None:
        r"""Configure all of the collected networks, and enable them."""
        networks = []
        explicit_disable = False
        network_map = {"gasnet": "gasnetex", "ucx": "ucx", "mpi": "mpi"}
        for py_attr, net_name in network_map.items():
            state = getattr(self, py_attr).state
            if state.enabled():
                networks.append(net_name)
            elif state.explicit:
                # explicitly disabled
                explicit_disable = True

        if len(networks) > 1:
            self.log_warning(
                "Building Realm with multiple networking backends "
                f"({', '.join(networks)}) is not fully supported currently.",
            )
        if networks:
            self.manager.set_cmake_variable(
                self.Legion_NETWORKS, ";".join(networks)
            )
        elif explicit_disable:
            # ensure that it is properly cleared
            self.manager.set_cmake_variable(self.Legion_NETWORKS, "")

    def configure(self) -> None:
        r"""Configure Legion."""
        super().configure()
        self.log_execute_func(self.configure_root_dirs)
        self.log_execute_func(self.configure_variables)
        self.log_execute_func(self.configure_cuda)
        self.log_execute_func(self.configure_gasnet)
        self.log_execute_func(self.configure_openmp)
        self.log_execute_func(self.configure_hdf5)
        self.log_execute_func(self.configure_llvm)
        self.log_execute_func(self.configure_python)
        self.log_execute_func(self.configure_zlib)
        self.log_execute_func(self.configure_networks)

    def summarize(self) -> str:
        r"""Summarize Legion.

        Returns
        -------
        summary : str
            A summary of configured Legion.
        """
        lines = []
        dir_group = self.DirGroup
        if root_dir := self.manager.get_cmake_variable(
            dir_group.Legion_ROOT  # type: ignore[attr-defined]
        ):
            lines.append(("Root dir", root_dir))
        elif root_dir := self.manager.get_cmake_variable(
            dir_group.CPM_Legion_SOURCE  # type: ignore[attr-defined]
        ):
            lines.append(("Root dir", root_dir))
        else:
            lines.append(("Downloaded", True))

        if cxx_flags := self.manager.get_cmake_variable(self.Legion_CXX_FLAGS):
            lines.append(("C++ flags", cxx_flags))

        if cuda := self.manager.get_cmake_variable(self.Legion_USE_CUDA):
            lines.append(("With CUDA", cuda))
            if cuda_flags := self.manager.get_cmake_variable(
                self.Legion_CUDA_FLAGS
            ):
                lines.append(("CUDA flags", cuda_flags))
            if cuda_arch := self.manager.get_cmake_variable(
                self.Legion_CUDA_ARCH
            ):
                lines.append(("CUDA arch", cuda_arch))

        if networks := self.manager.get_cmake_variable(self.Legion_NETWORKS):
            pass
        else:
            networks = "None"
        lines.append(("Networks", networks))

        # TODO continue

        lines.extend(
            [
                ("Bounds checks", self.cl_args.legion_bounds_checks.value),
                ("Max dim", self.cl_args.legion_max_dim.value),
                ("Max fields", self.cl_args.legion_max_fields.value),
                ("Build Spy", self.cl_args.legion_spy.value),
                (
                    "Build Rust profiler",
                    self.cl_args.legion_rust_profiler.value,
                ),
            ]
        )
        return self.create_package_summary(lines)


def create_package(manager: ConfigurationManager) -> Legion:
    return Legion(manager)
