# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

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
from ...util.exception import UnsatisfiableConfigurationError
from ...util.utility import dest_to_flag
from ..package import Package
from .cuda import CUDA
from .gasnet import GASNet
from .mpi import MPI
from .openmp import OpenMP
from .python import Python
from .ucx import UCX
from .zlib import ZLIB

if TYPE_CHECKING:
    from ...manager import ConfigurationManager


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
                dest="with_legion_src_dir",
                type=Path,
                help="Path to an existing Legion source directory.",
            ),
            cmake_var=CMAKE_VARIABLE("CPM_Legion_SOURCE", CMakePath),
        ),
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
            default=6,
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

    Legion_EMBED_GASNet_CONFIGURE_ARGS: Final = CMAKE_VARIABLE(
        "Legion_EMBED_GASNet_CONFIGURE_ARGS", CMakeList
    )
    Legion_USE_CUDA: Final = CMAKE_VARIABLE("Legion_USE_CUDA", CMakeBool)
    Legion_USE_OpenMP: Final = CMAKE_VARIABLE("Legion_USE_OpenMP", CMakeBool)
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
    CPM_DOWNLOAD_Legion: Final = CMAKE_VARIABLE(
        "CPM_DOWNLOAD_Legion", CMakeBool
    )
    Legion_DIR: Final = CMAKE_VARIABLE("Legion_DIR", CMakePath)

    def __init__(self, manager: ConfigurationManager) -> None:
        r"""Construct a Legion Package.

        Parameters
        ----------
        manager : ConfigurationManager
            The configuration manager to manage this package.
        """
        super().__init__(
            manager=manager,
            name="Legion",
            always_enabled=True,
            dependencies=(CUDA, GASNet, OpenMP, Python, MPI, UCX, ZLIB),
        )

    def check_conflicting_options(self) -> None:
        r"""Check for conflicting options that are too complicated to "
        "describe statically.

        Raises
        ------
        UnsatisfiableConfigurationError
            If both --with-legion-src-dir and --legion-branch are set
        """
        with_legion_src_dir = self.cl_args.with_legion_src_dir
        legion_branch = self.cl_args.legion_branch
        if with_legion_src_dir.value and legion_branch.value:
            msg = (
                "Cannot specify both "
                f"{dest_to_flag(with_legion_src_dir.name)} and "
                f"{dest_to_flag(legion_branch.name)}, their combined meaning "
                "is ambiguous. If the source dir is given, the contents of "
                "the directory are used as-is (i.e. using whatever branch or "
                "commit that dir is currently on), so the chosen Legion "
                "branch would have no effect."
            )
            raise UnsatisfiableConfigurationError(msg)

    def configure_root_dirs(self) -> None:
        r"""Configure the various "root" directories that Legion requires."""
        dir_group = self.DirGroup
        if (lg_dir := self.cl_args.legion_dir).cl_set:
            self.manager.set_cmake_variable(
                dir_group.Legion_ROOT,  # type: ignore[attr-defined]
                lg_dir.value,
            )
        elif (lg_src_dir := self.cl_args.with_legion_src_dir).cl_set:
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
        cuda_state = self.deps.CUDA.state
        if cuda_state.enabled():
            self.manager.set_cmake_variable(self.Legion_USE_CUDA, True)
            self.append_flags_if_set(
                self.Legion_CUDA_FLAGS, self.cl_args.legion_cuda_flags
            )
        elif self.cl_args.legion_cuda_flags.cl_set:
            msg = (
                "--legion-cuda-flags set "
                f"({self.cl_args.legion_cuda_flags.value}), "
                "but CUDA is not enabled."
            )
            raise RuntimeError(msg)
        elif cuda_state.explicitly_disabled():
            self.manager.set_cmake_variable(self.Legion_USE_CUDA, False)

    def configure_gasnet(self) -> None:
        r"""Configure Legion to use GASNet. Does nothing if GASNet is not
        enabled.
        """
        if self.deps.GASNet.state.enabled():
            self.manager.append_cmake_variable(
                self.Legion_EMBED_GASNet_CONFIGURE_ARGS,
                ["--with-ibv-max-hcas=8"],
            )

    def configure_openmp(self) -> None:
        r"""Configure Legion to use OpenMP. Does nothing if OpenMP is not
        enabled.
        """
        omp_state = self.deps.OpenMP.state
        if omp_state.enabled():
            self.manager.set_cmake_variable(self.Legion_USE_OpenMP, True)
        elif omp_state.explicitly_disabled():
            self.manager.set_cmake_variable(self.Legion_USE_OpenMP, False)

    def configure_python(self) -> None:
        r"""Configure Legion to use Python. Does nothing if Python is not
        enabled.
        """
        python = cast(Python, self.deps.Python)
        py_state = python.state
        if py_state.enabled():
            self.manager.set_cmake_variable(self.Legion_BUILD_BINDINGS, True)
            self.manager.set_cmake_variable(self.Legion_USE_Python, True)
            self.manager.set_cmake_variable(self.Legion_BUILD_JUPYTER, True)
            self.manager.set_cmake_variable(
                self.Legion_Python_Version, python.lib_version
            )
        elif py_state.explicitly_disabled():
            self.manager.set_cmake_variable(self.Legion_BUILD_BINDINGS, False)
            self.manager.set_cmake_variable(self.Legion_BUILD_JUPYTER, False)
            self.manager.set_cmake_variable(self.Legion_USE_Python, False)

    def configure_zlib(self) -> None:
        r"""Configure Legion to use ZLIB. Disables ZLIB if is not enabled."""
        zlib_state = self.deps.ZLIB.state
        if zlib_state.enabled():
            self.manager.set_cmake_variable(self.Legion_USE_ZLIB, True)
        elif zlib_state.explicitly_disabled():
            self.manager.set_cmake_variable(self.Legion_USE_ZLIB, False)

    def configure_networks(self) -> None:
        r"""Configure all of the collected networks, and enable them."""
        networks = []
        explicit_disable = False
        network_map = {"GASNet": "gasnetex", "UCX": "ucx", "MPI": "mpi"}
        for py_attr, net_name in network_map.items():
            state = getattr(self.deps, py_attr).state
            if state.enabled():
                networks.append(net_name)
            elif state.explicit:
                # explicitly disabled
                explicit_disable = True

        if len(networks) > 1:
            self.log_warning(
                "Building Realm with multiple networking backends "
                f"({', '.join(networks)}) is not fully supported currently."
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
        self.log_execute_func(self.check_conflicting_options)
        self.log_execute_func(self.configure_root_dirs)
        self.log_execute_func(self.configure_variables)
        self.log_execute_func(self.configure_cuda)
        self.log_execute_func(self.configure_gasnet)
        self.log_execute_func(self.configure_openmp)
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
        m = self.manager

        def get_location() -> Path | None:
            dir_group = self.DirGroup
            root_dir = m.get_cmake_variable(
                dir_group.Legion_ROOT  # type: ignore[attr-defined]
            )
            if root_dir:
                return Path(root_dir)

            root_dir = m.get_cmake_variable(self.Legion_DIR)
            if root_dir:
                return Path(root_dir)

            root_dir = m.get_cmake_variable(
                dir_group.CPM_Legion_SOURCE  # type: ignore[attr-defined]
            )
            if root_dir:
                root_path = Path(root_dir)
                # If the source directory is relative to the cmake
                # directory, then we downloaded Legion, but set
                # CPM_Legion_Source ourselves.
                if not root_path.is_relative_to(m.project_cmake_dir):
                    return root_path
            return None

        lines: list[tuple[str, Any]] = []
        root_dir = get_location()
        downloaded = root_dir is None
        lines.append(("Downloaded", downloaded))
        if not downloaded:
            assert root_dir is not None  # pacify mypy
            lines.append(("  Root dir", root_dir))

        if cxx_flags := m.get_cmake_variable(self.Legion_CXX_FLAGS):
            lines.append(("C++ flags", cxx_flags))

        lines.append(("With CUDA", m.get_cmake_variable(self.Legion_USE_CUDA)))
        if cuda_flags := m.get_cmake_variable(self.Legion_CUDA_FLAGS):
            lines.append(("CUDA flags", cuda_flags))

        if networks := m.get_cmake_variable(self.Legion_NETWORKS):
            pass
        else:
            networks = "None"
        lines.append(("Networks", networks))
        lines.append(
            ("Bounds checks", m.get_cmake_variable(self.Legion_BOUNDS_CHECKS))
        )
        lines.append(("Max dim", m.get_cmake_variable(self.Legion_MAX_DIM)))
        lines.append(
            ("Max fields", m.get_cmake_variable(self.Legion_MAX_FIELDS))
        )
        lines.append(("Build Spy", m.get_cmake_variable(self.Legion_SPY)))
        lines.append(
            (
                "Build Rust profiler",
                m.get_cmake_variable(self.Legion_BUILD_RUST_PROFILER),
            )
        )
        # TODO continue

        return self.create_package_summary(lines)


def create_package(manager: ConfigurationManager) -> Legion:
    return Legion(manager)
