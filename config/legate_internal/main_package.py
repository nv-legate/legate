# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import re
import sys
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

from aedifix import (
    CMAKE_VARIABLE,
    ArgSpec,
    CMakeBool,
    CMakeExecutable,
    CMakeList,
    CMakeString,
    ConfigArgument,
    ConfigurationManager,
    MainPackage,
)
from aedifix.packages.cmake import CMake
from aedifix.packages.cuda import CUDA
from aedifix.packages.python import Python

from .packages.gasnet import GASNet
from .packages.hdf5 import HDF5
from .packages.legion import Legion
from .packages.mpi import MPI
from .packages.nccl import NCCL
from .packages.realm import Realm
from .packages.ucx import UCX

if TYPE_CHECKING:
    from collections.abc import Sequence


class Legate(MainPackage):
    name = "Legate"

    dependencies = (
        CMake,
        Realm,
        Legion,
        Python,
        HDF5,
        GASNet,
        NCCL,
        UCX,
        CUDA,
        MPI,
    )

    legate_BUILD_DOCS: Final = ConfigArgument(
        name="--with-docs",
        spec=ArgSpec(
            dest="with_docs", type=bool, help="Generate docs build makefile"
        ),
        cmake_var=CMAKE_VARIABLE("legate_BUILD_DOCS", CMakeBool),
    )
    legate_BUILD_TESTS: Final = ConfigArgument(
        name="--with-tests",
        spec=ArgSpec(dest="with_tests", type=bool, help="Build tests"),
        cmake_var=CMAKE_VARIABLE("legate_BUILD_TESTS", CMakeBool),
    )
    legate_BUILD_BENCHMARKS: Final = ConfigArgument(
        name="--with-benchmarks",
        spec=ArgSpec(
            dest="with_benchmarks", type=bool, help="Build benchmarks"
        ),
        cmake_var=CMAKE_VARIABLE("legate_BUILD_BENCHMARKS", CMakeBool),
    )
    legate_CXX_FLAGS: Final = ConfigArgument(
        name="--legate-cxx-flags",
        spec=ArgSpec(
            dest="legate_cxx_flags", nargs=1, help="C++ flags for Legate"
        ),
        cmake_var=CMAKE_VARIABLE("legate_CXX_FLAGS", CMakeList),
    )
    legate_CUDA_FLAGS: Final = ConfigArgument(
        name="--legate-cuda-flags",
        spec=ArgSpec(
            dest="legate_cuda_flags", nargs=1, help="CUDA flags for Legate"
        ),
        cmake_var=CMAKE_VARIABLE("legate_CUDA_FLAGS", CMakeList),
    )
    legate_LINKER_FLAGS: Final = ConfigArgument(
        name="--legate-linker-flags",
        spec=ArgSpec(
            dest="legate_linker_flags", nargs=1, help="Linker flags for Legate"
        ),
        cmake_var=CMAKE_VARIABLE("legate_LINKER_FLAGS", CMakeList),
    )
    BUILD_MARCH: Final = ConfigArgument(
        name="--build-march",
        spec=ArgSpec(
            dest="build_march",
            default="native",
            help="CPU architecture to build for",
        ),
        cmake_var=CMAKE_VARIABLE("BUILD_MARCH", CMakeString),
    )
    LEGATE_CLANG_TIDY: Final = ConfigArgument(
        name="--clang-tidy-executable",
        spec=ArgSpec(
            dest="clang_tidy_executable",
            type=Path,
            default=shutil.which("clang-tidy"),
            help="clang-tidy executable",
        ),
        cmake_var=CMAKE_VARIABLE("LEGATE_CLANG_TIDY", CMakeExecutable),
    )
    # Enable stub fatbins to allow running clang-tidy without building
    # fatbins/Legion during analysis.
    legate_FAKE_FATBINS_FOR_TIDY: Final = ConfigArgument(
        name="--with-fake-fatbins-for-tidy",
        spec=ArgSpec(
            dest="with_fake_fatbins_for_tidy",
            type=bool,
            help=(
                "Emit stub fatbins for clang-tidy "
                "(avoids building CUDA/Legion during tidy)"
            ),
        ),
        cmake_var=CMAKE_VARIABLE("legate_FAKE_FATBINS_FOR_TIDY", CMakeBool),
    )
    legate_LEGION_REPOSITORY: Final = CMAKE_VARIABLE(
        "legate_LEGION_REPOSITORY", CMakeString
    )
    legate_LEGION_BRANCH: Final = CMAKE_VARIABLE(
        "legate_LEGION_BRANCH", CMakeString
    )
    legate_ENABLE_SANITIZERS: Final = CMAKE_VARIABLE(
        "legate_ENABLE_SANITIZERS", CMakeBool
    )
    legate_IGNORE_INSTALLED_PACKAGES: Final = ConfigArgument(
        name="--ignore-installed-packages",
        spec=ArgSpec(
            dest="ignore_installed_packages",
            type=bool,
            default=True,
            help=(
                "If true, when deciding to search for, or download third-party"
                " packages, never search and always download. WARNING: "
                "setting this option to false may make your builds "
                "non-idempotent! Prior builds (and installations) may affect "
                "the current ones in non-trivial ways. reconfiguring may "
                "yield different results."
            ),
        ),
        cmake_var=CMAKE_VARIABLE(
            "legate_IGNORE_INSTALLED_PACKAGES", CMakeBool
        ),
    )
    legate_USE_CPROFILE: Final = ConfigArgument(
        name="--with-cprofile",
        spec=ArgSpec(
            dest="use_cprofile",
            type=bool,
            help="If true, Legate will be built with detailed cProfile output."
            " In particular, this flag will enable profiling Cython code"
            " in Legate. WARNING: When enabled, this may negatively affect"
            " program performance. cProfile is a Python built-in module"
            " for profiling runtime performance, measuring function"
            " calls and execution time in Python programs.",
        ),
        cmake_var=CMAKE_VARIABLE("legate_USE_CPROFILE", CMakeBool),
    )

    legate_USE_HDF5: Final = CMAKE_VARIABLE("legate_USE_HDF5", CMakeBool)
    legate_USE_HDF5_VFD_GDS: Final = ConfigArgument(
        name="--with-hdf5-vfd-gds",
        spec=ArgSpec(
            dest="with_hdf5_vfd_gds",
            type=bool,
            help=(
                "Enable VFD GPU Direct Storage support in Legate IO. Support "
                "for this is automatically detected based on the availability "
                "of both CUDA and HDF5."
            ),
        ),
        cmake_var=CMAKE_VARIABLE("legate_USE_HDF5_VFD_GDS", CMakeBool),
    )
    legate_USE_GASNET: Final = CMAKE_VARIABLE("legate_USE_GASNET", CMakeBool)
    legate_USE_NCCL: Final = CMAKE_VARIABLE("legate_USE_NCCL", CMakeBool)
    legate_USE_UCX: Final = CMAKE_VARIABLE("legate_USE_UCX", CMakeBool)
    legate_USE_CUDA: Final = CMAKE_VARIABLE("legate_USE_CUDA", CMakeBool)
    legate_USE_MPI: Final = CMAKE_VARIABLE("legate_USE_MPI", CMakeBool)
    legate_BUILD_MPI_WRAPPER: Final = ConfigArgument(
        name="--with-mpi-wrapper",
        spec=ArgSpec(
            dest="with_mpi_wrapper",
            type=bool,
            help="Build Legate's MPI wrapper shared libraries.",
        ),
        cmake_var=CMAKE_VARIABLE("legate_BUILD_MPI_WRAPPER", CMakeBool),
    )

    def __init__(
        self, manager: ConfigurationManager, argv: Sequence[str]
    ) -> None:
        r"""Construct a Legate main package.

        Parameters
        ----------
        manager : ConfigurationManager
            The configuration manager to manage this package.
        argv : Sequence[str]
            The command line arguments for this configuration.
        """
        from scripts.get_legate_dir import get_legate_dir  # noqa: PLC0415

        legate_dir = Path(get_legate_dir())
        super().__init__(
            manager=manager,
            argv=argv,
            arch_name="LEGATE_ARCH",
            project_dir_name="LEGATE_DIR",
            project_dir_value=legate_dir,
            project_src_dir=legate_dir / "src",
            default_arch_file_path=(
                legate_dir / "scripts" / "get_legate_arch.py"
            ),
        )

    @classmethod
    def from_argv(
        cls, manager: ConfigurationManager, argv: Sequence[str]
    ) -> Legate:
        r"""Construct a Legate main package from argv.

        Parameters
        ----------
        manager : ConfigurationManager
            The configuration manager to manage this package.
        argv : Sequence[str]
            The command line arguments for this configuration

        Returns
        -------
        package : Legate
            The constructed main package.
        """
        return cls(manager, argv)

    def maybe_uninstall_legate(self) -> None:
        r"""Uninstall Legate if --with-clean is given on command line
        arguments.
        """
        # Returns all the packages in the format:
        #
        # Package Version Editable project location
        # ------- ------- -------------------------
        # foo     0.7.16
        # bar     2.4.1
        # baz     2.15.0  /path/to/baz
        # ...
        try:
            import pip  # noqa: F401, PLC0415
        except ModuleNotFoundError as mnfe:
            self.log(
                f"pip does not appear to be installed: '{mnfe}'. Nothing to do"
            )
            return

        installed_packages = self.log_execute_command(
            [
                sys.executable,
                "-m",
                "pip",
                "--disable-pip-version-check",
                "list",
            ]
        ).stdout.splitlines()
        # skip the "Package Version" header and divider lines
        installed_packages = installed_packages[2:]
        package_names = (
            line.split(maxsplit=1)[0] for line in installed_packages if line
        )
        found_legate = any(name.startswith("legate") for name in package_names)
        self.log(f"Have pre-existing legate installation: {found_legate}")

        if not found_legate:
            return

        if self.cl_args.with_clean.value:
            cmd = [sys.executable, "-m", "pip", "uninstall", "--yes", "legate"]
            str_cmd = " ".join(cmd)
            self.log_warning(
                f"Running {str_cmd!r} to uninstall legate as part of a clean "
                "build."
            )
            self.log_execute_command(cmd)
        else:
            self.log(
                "No clean requested, leaving potentially installed legate "
                "in place"
            )
            self.log_warning(
                "You appear to have previously installed Legate, which "
                "may interfere with the current and/or future "
                "(re-)configurations of Legate. Issues stemming from "
                "this are likely to manifest at build-time, not "
                "configure-time, and so if you encounter confusing build "
                "errors the culprit is likely this.\n"
                "\n"
                "The user is strongly encouranged to run either:\n"
                "\n"
                f"$ {sys.executable} -m pip uninstall --yes legate\n"
                "\n"
                "(then retry configuration), or, re-run configuration "
                f"with the {self.WITH_CLEAN.name} flag."
            )

    def setup(self) -> None:
        r"""Setup Legate."""
        self.log_execute_func(self.maybe_uninstall_legate)
        super().setup()

    def check_min_cmake_version(self) -> None:
        r"""Assert the minimum cmake version is met."""
        try:
            from packaging.version import (  # noqa: PLC0415
                parse as version_parse,
            )
        except ModuleNotFoundError:
            # error: All conditional function variants must have identical
            # signatures
            #
            # Yes, I know, but this is good enough.
            def version_parse(  # type: ignore[misc]
                version: str,
            ) -> tuple[int, ...]:
                args = (a.strip() for a in version.split("."))
                return tuple(int(a) for a in args if a)

        min_ver_re = re.compile(
            r"cmake_minimum_required\(.*VERSION\s+([\d\.]+)"
        )
        cmakelists_txt = self.project_src_dir / "CMakeLists.txt"
        with cmakelists_txt.open() as fd:
            for line in fd:
                if re_match := min_ver_re.search(line):
                    min_ver = re_match.group(1)
                    break
            else:
                msg = (
                    "Failed to parse minimum required CMake version from"
                    f" {cmakelists_txt}"
                )
                raise RuntimeError(msg)

        self.log(f"Minimum cmake version required: {min_ver}")
        cmake = cast(CMake, self.deps.CMake)
        if version_parse(cmake.version) < version_parse(min_ver):
            cmake_exe = self.manager.get_cmake_variable(cmake.CMAKE_COMMAND)
            msg = (
                f"CMake executable {cmake_exe} too old! Expected version "
                f"{min_ver}, have {cmake.version}"
            )
            raise RuntimeError(msg)

    def configure_legate_variables(self) -> None:
        r"""Configure the general variables for Legate."""
        self.append_flags_if_set(
            self.legate_CXX_FLAGS, self.cl_args.legate_cxx_flags
        )
        self.append_flags_if_set(
            self.legate_LINKER_FLAGS, self.cl_args.legate_linker_flags
        )
        self.append_flags_if_set(
            self.legate_CUDA_FLAGS, self.cl_args.legate_cuda_flags
        )
        self.set_flag_if_user_set(
            self.legate_BUILD_DOCS, self.cl_args.with_docs
        )
        self.set_flag_if_user_set(
            self.legate_BUILD_TESTS, self.cl_args.with_tests
        )
        self.set_flag_if_user_set(
            self.legate_BUILD_BENCHMARKS, self.cl_args.with_benchmarks
        )
        self.set_flag_if_user_set(self.BUILD_MARCH, self.cl_args.build_march)
        self.set_flag_if_user_set(
            self.legate_BUILD_MPI_WRAPPER, self.cl_args.with_mpi_wrapper
        )
        build_type = self.cl_args.build_type
        if "sanitizer" in build_type.value:
            self.manager.set_cmake_variable(
                self.legate_ENABLE_SANITIZERS, True
            )
        elif build_type.cl_set:
            self.manager.set_cmake_variable(
                self.legate_ENABLE_SANITIZERS, False
            )
        ignore_packages = self.cl_args.ignore_installed_packages
        if (not ignore_packages.value) and ignore_packages.cl_set:
            flag_name = ignore_packages.name.replace("_", "-")
            self.log_warning(
                f"Setting --{flag_name} to false may make your builds "
                "non-idempotent! Prior builds (and installations) may affect "
                "the current one in non-trivial ways."
                "\n"
                "\n"
                "** If you are a developer, building a development build, "
                "this is probably not what you want. Please consider removing "
                "this flag from your command-line arguments. **"
                "\n"
                "\n"
                "For example, consider the following:"
                "\n"
                "\n"
                f" 1. ./configure --{flag_name}=0 --with-foo (CMake downloads and builds libfoo.so)\n"  # noqa: E501
                f" 2. pip install . (CMake -- as a byproduct of installing {self.project_name} -- installs libfoo.so)\n"  # noqa: E501
                " 3. ./reconfigure... (CMake now picks up installed libfoo.so instead of reusing the downloaded one)\n"  # noqa: E501
                "\n"
                "The package can now no longer be built."
                "\n"
                "\n"
                "CMake still has a local target libfoo.so (from step 1), but "
                "now due to step 3, libfoo.so is considered 'imported' "
                "(because CMake found the installed version first). Imported "
                "packages provide no recipes to build their products "
                "(libfoo.so) and so the build is broken."
            )
        self.set_flag(self.legate_IGNORE_INSTALLED_PACKAGES, ignore_packages)

    def _warn_legion_patches(self, src_dir: Path) -> None:
        patch_dir = self.project_src_dir / "cmake" / "patches"
        assert patch_dir.is_dir()
        patches = [f"- {p}" for p in patch_dir.glob("legion*")]

        if not patches:
            # We could do an assert, but then if we do end up removing all
            # patches this will needlessly break. So more robust simply to
            # bail.
            self.log(
                "No patches to apply to legion, no need to warn about src-dir"
            )
            return

        plist = "\n".join(patches)
        self.log_warning(
            "You have provided a source directory for Legion "
            f"({src_dir}). Legate requires that a series of patches are "
            "applied to Legion. This is performed automatically by the "
            "build-system EXCEPT when a source directory is provided. "
            "This is a limitation of the current build system and will be "
            "fixed in a future release.\n"
            "\n"
            "You must manually apply the patches:\n"
            "\n"
            f"{plist}\n"
            "\n"
            "before continuing."
        )

    def configure_legion(self) -> None:
        r"""Configure Legion for use with Legate."""
        self.set_flag_if_user_set(
            self.legate_LEGION_BRANCH, self.cl_args.legion_branch
        )

        legion = cast(Legion, self.deps.Legion)

        if src_dir := self.manager.get_cmake_variable(
            legion.DirGroup.CPM_Legion_SOURCE  # type: ignore[attr-defined]
        ):
            self._warn_legion_patches(src_dir)

    def configure_clang_tidy(self) -> None:
        r"""Configure clang-tidy variables."""
        self.set_flag_if_user_set(
            self.LEGATE_CLANG_TIDY, self.cl_args.clang_tidy_executable
        )
        # Allow explicit CLI control for fake fatbins for tidy
        self.set_flag_if_user_set(
            self.legate_FAKE_FATBINS_FOR_TIDY,
            self.cl_args.with_fake_fatbins_for_tidy,
        )

    def configure_cprofile(self) -> None:
        r"""Configure cprofile variables."""
        self.set_flag_if_user_set(
            self.legate_USE_CPROFILE, self.cl_args.use_cprofile
        )

    def configure_hdf5(self) -> None:
        r"""Configure HDF5 variables."""
        hdf5_state = self.deps.HDF5.state
        if hdf5_state.enabled():
            self.manager.set_cmake_variable(self.legate_USE_HDF5, True)
        elif hdf5_state.explicitly_disabled():
            self.manager.set_cmake_variable(self.legate_USE_HDF5, False)

        self.set_flag_if_user_set(
            self.legate_USE_HDF5_VFD_GDS, self.cl_args.with_hdf5_vfd_gds
        )

    def configure_gasnet(self) -> None:
        r"""Configure GASNet variables."""
        state = self.deps.GASNet.state
        if state.enabled():
            self.manager.set_cmake_variable(self.legate_USE_GASNET, True)
        elif state.explicitly_disabled():
            self.manager.set_cmake_variable(self.legate_USE_GASNET, False)

    def configure_nccl(self) -> None:
        r"""Configure NCCL variables."""
        state = self.deps.NCCL.state
        if state.enabled():
            self.manager.set_cmake_variable(self.legate_USE_NCCL, True)
        elif state.explicitly_disabled():
            self.manager.set_cmake_variable(self.legate_USE_NCCL, False)

    def configure_ucx(self) -> None:
        r"""Configure UCX variables."""
        state = self.deps.UCX.state
        if state.enabled():
            self.manager.set_cmake_variable(self.legate_USE_UCX, True)
        elif state.explicitly_disabled():
            self.manager.set_cmake_variable(self.legate_USE_UCX, False)

    def configure_cuda(self) -> None:
        r"""Configure CUDA variables."""
        state = self.deps.CUDA.state
        if state.enabled():
            self.manager.set_cmake_variable(self.legate_USE_CUDA, True)
        elif state.explicitly_disabled():
            self.manager.set_cmake_variable(self.legate_USE_CUDA, False)

    def configure_mpi(self) -> None:
        r"""Configure CUDA variables."""
        state = self.deps.MPI.state
        if state.enabled():
            self.manager.set_cmake_variable(self.legate_USE_MPI, True)
        elif state.explicitly_disabled():
            self.manager.set_cmake_variable(self.legate_USE_MPI, False)

    def configure(self) -> None:
        r"""Configure Legate."""
        super().configure()
        self.log_execute_func(self.check_min_cmake_version)
        self.log_execute_func(self.configure_legate_variables)
        self.log_execute_func(self.configure_legion)
        self.log_execute_func(self.configure_clang_tidy)
        self.log_execute_func(self.configure_cprofile)
        self.log_execute_func(self.configure_hdf5)
        self.log_execute_func(self.configure_gasnet)
        self.log_execute_func(self.configure_nccl)
        self.log_execute_func(self.configure_ucx)
        self.log_execute_func(self.configure_cuda)
        self.log_execute_func(self.configure_mpi)

    def _summarize_flags(self) -> list[tuple[str, Any]]:
        def make_summary(
            name: str, cmake_varname: ConfigArgument
        ) -> tuple[str, str]:
            flags = self.manager.get_cmake_variable(cmake_varname)
            match flags:
                case list() | tuple():
                    flags_str = " ".join(flags)
                case None:
                    flags_str = ""
                case str():
                    flags_str = flags
                case _:
                    raise TypeError(type(flags))
            return (f"{name} Flags", flags_str.replace(";", " "))

        return [
            make_summary("C++", self.legate_CXX_FLAGS),
            make_summary("Linker", self.legate_LINKER_FLAGS),
            make_summary("CUDA", self.legate_CUDA_FLAGS),
        ]

    def _summarize_python(self) -> list[tuple[str, Any]]:
        return [("Python bindings", self.deps.Python.state.enabled())]

    def _summarize_misc(self) -> list[tuple[str, Any]]:
        m = self.manager
        return [
            ("Tests", m.get_cmake_variable(self.legate_BUILD_TESTS)),
            ("Docs", m.get_cmake_variable(self.legate_BUILD_DOCS)),
            ("Benchmarks", m.get_cmake_variable(self.legate_BUILD_BENCHMARKS)),
            ("HDF5", m.get_cmake_variable(self.legate_USE_HDF5)),
            (
                "HDF5 VFD GDS",
                m.get_cmake_variable(self.legate_USE_HDF5_VFD_GDS),
            ),
            ("GASNet", m.get_cmake_variable(self.legate_USE_GASNET)),
            ("NCCL", m.get_cmake_variable(self.legate_USE_NCCL)),
            ("UCX", m.get_cmake_variable(self.legate_USE_UCX)),
            ("CUDA", m.get_cmake_variable(self.legate_USE_CUDA)),
            ("MPI", m.get_cmake_variable(self.legate_USE_MPI)),
            (
                "MPI wrapper",
                m.get_cmake_variable(self.legate_BUILD_MPI_WRAPPER),
            ),
        ]

    def summarize(self) -> str:
        r"""Summarize Legate.

        Returns
        -------
        summary : str
            The summary of Legate.
        """
        lines = []
        for summarizer in (
            self._summarize_flags,
            self._summarize_misc,
            self._summarize_python,
        ):
            lines.extend(summarizer())
        return self.create_package_summary(lines)
