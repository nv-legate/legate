# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

import enum
import multiprocessing as mp
import os
import platform
import shlex
import shutil
import textwrap
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final

from ..cmake.cmake_flags import (
    CMAKE_VARIABLE,
    CMakeBool,
    CMakeExecutable,
    CMakeInt,
    CMakeList,
    CMakePath,
    CMakeString,
)
from ..util.argument_parser import ArgSpec, ConfigArgument, _str_to_bool
from ..util.utility import ValueProvenance, flag_to_dest
from .package import Package

if TYPE_CHECKING:
    from ..manager import ConfigurationManager

_DEFAULT_BUILD_TYPE: Final = os.environ.get(
    "CMAKE_BUILD_TYPE", "release"
).casefold()
_CMAKE_BUILD_TYPE_MAP: Final = {
    "debug": "Debug",
    "release": "Release",
    "release-debug": "RelWithDebInfo",
    # An alias for release-debug
    "relwithdebinfo": "RelWithDebInfo",
    # This still maps to Debug because we don't want to invent a new build type
    # for it. Specifically, we want the main package to be both debug and
    # sanitized, but have all other packages be regular debug builds. It is up
    # to the main package to properly set sanitizer flags for itself based on
    # the --build-type command line argument.
    "debug-sanitizer": "Debug",
}


def _make_default_flags() -> dict[str, dict[str, list[str]]]:
    def to_cuda_flags(flags: list[str]) -> list[str]:
        return [f"--compiler-options={f}" for f in map(shlex.quote, flags)]

    def make_subdict(
        c_flags: list[str],
        cxx_flags: list[str] | None = None,
        cuda_flags: list[str] | None = None,
    ) -> dict[str, list[str]]:
        if cxx_flags is None:
            cxx_flags = c_flags[:]
        if cuda_flags is None:
            cuda_flags = to_cuda_flags(cxx_flags)

        return {
            "CFLAGS": c_flags,
            "CXXFLAGS": cxx_flags,
            "CUDAFLAGS": cuda_flags,
        }

    debug_c_flags = ["-O0", "-g", "-g3"]
    debug_cuda_flags = ["-g"] + to_cuda_flags(debug_c_flags)

    release_c_flags = ["-O3"]

    reldeb_c_flags = debug_c_flags + release_c_flags
    reldeb_cuda_flags = ["-g"] + to_cuda_flags(reldeb_c_flags)

    return {
        "Debug": make_subdict(
            c_flags=debug_c_flags, cuda_flags=debug_cuda_flags
        ),
        "Release": make_subdict(c_flags=release_c_flags),
        "RelWithDebInfo": make_subdict(
            c_flags=reldeb_c_flags, cuda_flags=reldeb_cuda_flags
        ),
    }


_DEFAULT_FLAGS: Final = _make_default_flags()
assert set(_CMAKE_BUILD_TYPE_MAP.values()) == set(_DEFAULT_FLAGS.keys())


class LibraryLinkage(str, enum.Enum):
    SHARED = "shared"
    STATIC = "static"

    def __str__(self) -> str:
        return self.name.casefold()


def _guess_c_compiler() -> str | None:
    for env_guess in ("CC", "CMAKE_C_COMPILER"):
        if guess := os.environ.get(env_guess):
            return guess
    if guess := shutil.which("cc"):
        return guess
    return None


def _guess_cxx_compiler() -> str | None:
    for env_guess in ("CXX", "CMAKE_CXX_COMPILER"):
        if guess := os.environ.get(env_guess):
            return guess
    for ccguess in ("c++", "C++", "CC", "CXX", "cxx"):
        if guess := shutil.which(ccguess):
            return guess
    return None


WITH_CLEAN_FLAG: Final = "--with-clean"
FORCE_FLAG: Final = "--force"
ON_ERROR_DEBUGGER_FLAG: Final = "--on-error-debugger"
DEBUG_CONFIGURE_FLAG: Final = "--debug-configure"


def _detect_num_cpus() -> int:
    if env_val := os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", ""):
        return int(env_val)
    return max(mp.cpu_count() - 1, 1)


DEFAULT_PROJECT_ARCH_SCRIPT_TEMPLATE: Final = textwrap.dedent(
    """
#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) {YEAR} NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# This file was automatically generated by:
# {FILE}
# Any modifications may be lost when configure is next invoked.
from __future__ import annotations


def get_{PROJECT_NAME}_arch() -> str:
    return "{PROJECT_ARCH_VALUE}"


def main() -> None:
    print(get_{PROJECT_NAME}_arch(), end="", flush=True)


if __name__ == "__main__":
    main()
""".strip()
)


class MainPackage(Package, ABC):
    DEBUG_CONFIGURE: Final = ConfigArgument(
        name=DEBUG_CONFIGURE_FLAG,
        spec=ArgSpec(
            dest=flag_to_dest(DEBUG_CONFIGURE_FLAG),
            type=int,
            default=0,
            const=1,
            nargs="?",
            help=(
                "Enable additional debugging flags to help debug configure. "
                "A higher value means more debug info."
            ),
        ),
        ephemeral=True,
    )
    ON_ERROR_DEBUGGER: Final = ConfigArgument(
        name=ON_ERROR_DEBUGGER_FLAG,
        spec=ArgSpec(
            dest=flag_to_dest(ON_ERROR_DEBUGGER_FLAG),
            type=bool,
            help=(
                "Start a post-mortem debugger if a Python exception was raised"
            ),
        ),
        ephemeral=True,
    )
    WITH_CLEAN: Final = ConfigArgument(
        name=WITH_CLEAN_FLAG,
        spec=ArgSpec(
            dest=flag_to_dest(WITH_CLEAN_FLAG),
            type=bool,
            help="Discard all existing configuration and start fresh",
        ),
        ephemeral=True,
    )
    FORCE: Final = ConfigArgument(
        name=FORCE_FLAG,
        spec=ArgSpec(
            dest=flag_to_dest(FORCE_FLAG),
            type=bool,
            help=(
                "Tell configure that you know what you are doing and force "
                "it to proceed, even if configure believes that doing so "
                "would be erroneous"
            ),
        ),
        ephemeral=True,
    )
    CMAKE_BUILD_PARALLEL_LEVEL: Final = ConfigArgument(
        name="--num-threads",
        spec=ArgSpec(
            dest="num_threads",
            type=int,
            nargs="?",
            default=_detect_num_cpus(),
            help="Number of threads with which to compile",
        ),
        cmake_var=CMAKE_VARIABLE("CMAKE_BUILD_PARALLEL_LEVEL", CMakeInt),
    )
    CMAKE_BUILD_TYPE: Final = ConfigArgument(
        name="--build-type",
        spec=ArgSpec(
            dest="build_type",
            choices=tuple(_CMAKE_BUILD_TYPE_MAP.keys()),
            default=_DEFAULT_BUILD_TYPE,
            help="Set the default build type",
        ),
        cmake_var=CMAKE_VARIABLE("CMAKE_BUILD_TYPE", CMakeString),
    )
    BUILD_SHARED_LIBS: Final = ConfigArgument(
        name="--library-linkage",
        spec=ArgSpec(
            dest="library_linkage",
            choices=(LibraryLinkage.SHARED, LibraryLinkage.STATIC),
            default=LibraryLinkage.SHARED,
            help="Set the default linkage strategy for built libraries",
        ),
        cmake_var=CMAKE_VARIABLE("BUILD_SHARED_LIBS", CMakeBool),
    )
    CMAKE_C_COMPILER: Final = ConfigArgument(
        name="--with-cc",
        spec=ArgSpec(
            dest="CC",
            type=Path,
            default=_guess_c_compiler(),
            help="Specify C compiler",
        ),
        cmake_var=CMAKE_VARIABLE("CMAKE_C_COMPILER", CMakeExecutable),
    )
    CMAKE_CXX_COMPILER: Final = ConfigArgument(
        name="--with-cxx",
        spec=ArgSpec(
            dest="CXX",
            type=Path,
            default=_guess_cxx_compiler(),
            help="Specify C++ compiler",
        ),
        cmake_var=CMAKE_VARIABLE("CMAKE_CXX_COMPILER", CMakeExecutable),
    )
    CMAKE_C_FLAGS: Final = ConfigArgument(
        name="--CFLAGS",
        spec=ArgSpec(dest="CFLAGS", nargs=1, help="C compiler flags"),
        cmake_var=CMAKE_VARIABLE("CMAKE_C_FLAGS", CMakeList),
    )
    CMAKE_CXX_FLAGS: Final = ConfigArgument(
        name="--CXXFLAGS",
        spec=ArgSpec(dest="CXXFLAGS", nargs=1, help="C++ compiler flags"),
        cmake_var=CMAKE_VARIABLE("CMAKE_CXX_FLAGS", CMakeList),
    )
    CMAKE_EXPORT_COMPILE_COMMANDS: Final = CMAKE_VARIABLE(
        "CMAKE_EXPORT_COMPILE_COMMANDS", CMakeBool
    )
    CMAKE_COLOR_DIAGNOSTICS: Final = CMAKE_VARIABLE(
        "CMAKE_COLOR_DIAGNOSTICS", CMakeBool
    )
    CMAKE_COLOR_MAKEFILE: Final = CMAKE_VARIABLE(
        "CMAKE_COLOR_MAKEFILE", CMakeBool
    )
    CMAKE_INSTALL_PREFIX: Final = ConfigArgument(
        name="--prefix",
        spec=ArgSpec(
            dest="prefix",
            type=Path,
            help=(
                "Default installation prefix. Defaults to /usr/local on Unix."
            ),
        ),
        cmake_var=CMAKE_VARIABLE("CMAKE_INSTALL_PREFIX", CMakePath),
    )

    __package_ignore_attrs__ = (
        "DEBUG_CONFIGURE",
        "ON_ERROR_DEBUGGER",
        "WITH_CLEAN",
        "FORCE",
        "CMAKE_BUILD_PARALLEL_LEVEL",
        "CMAKE_BUILD_TYPE",
        "BUILD_SHARED_LIBS",
        "CMAKE_C_COMPILER",
        "CMAKE_CXX_COMPILER",
        "CMAKE_C_FLAGS",
        "CMAKE_CXX_FLAGS",
        "CMAKE_EXPORT_COMPILE_COMMANDS",
        "CMAKE_COLOR_DIAGNOSTICS",
        "CMAKE_COLOR_MAKEFILE",
        "CMAKE_INSTALL_PREFIX",
    )

    __slots__ = (
        "_arch_name",
        "_arch_value",
        "_arch_value_provenance",
        "_proj_dir_name",
        "_proj_dir_value",
        "_proj_src_dir",
        "_default_arch_file_path",
    )

    def __init__(
        self,
        manager: ConfigurationManager,
        name: str,
        argv: Sequence[str],
        arch_name: str,
        project_dir_name: str,
        project_dir_value: Path,
        project_src_dir: Path | None = None,
        default_arch_file_path: Path | None = None,
    ) -> None:
        r"""Construct the MainPackage.

        Parameters
        ----------
        manager : ConfigurationManager
            The configuration manager that will manage the main package.
        name : str
            The name of the main package, e.g. 'Legate'.
        argv : Sequence[str]
            The command line options.
        arch_name : str
            The name of the arch value, e.g. 'LEGATE_ARCH'.
        project_dir_name : str
            The name of the project dir variable, e.g. 'LEGATE_DIR'.
        project_dir_value : Path
            The value of the project dir, e.g. /path/to/legate.internal.
        project_src_dir : Path, optional
            The path to the projects source directory for CMake. If not
            provided, ``project_dir_value`` is used instead.
        default_arch_file_path : Path, optional
            The location to place a file containing the default PROJECT_ARCH
            value. If not provided, or None, no file is emitted.

        Raises
        ------
        AssertionError
            If the project arch value does not start with a '-', ends with a
            '_', or does not end with 'ARCH', or is not all caps.
        ValueError
            If the project arch value is set (either from command line or
            environment variable) but empty.
        """
        super().__init__(manager=manager, name=name)
        assert not arch_name.startswith("-")
        assert not arch_name.endswith("_")
        assert arch_name.isupper()
        assert arch_name.endswith("ARCH")
        self._arch_name = arch_name
        self._arch_value, self._arch_value_provenance = (
            self.preparse_arch_value(argv)
        )
        if not self.arch_value:
            raise ValueError(
                f"WARNING: {self.arch_name} is set, but empty (set via "
                f"{self.arch_value_provenance})! This is extremely dangerous, "
                f"and WILL cause many options (e.g. {self.WITH_CLEAN.name}) "
                "to misbehave! Please set this to a non-empty value before"
                "continuing."
            )
        self._proj_dir_name = project_dir_name
        self._proj_dir_value = project_dir_value.resolve(strict=True)
        if project_src_dir is None:
            project_src_dir = self._proj_dir_value
        self._proj_src_dir = project_src_dir.resolve(strict=True)
        self._default_arch_file_path = default_arch_file_path

    @classmethod
    @abstractmethod
    def from_argv(
        cls, manager: ConfigurationManager, argv: Sequence[str]
    ) -> MainPackage:
        raise NotImplementedError()

    @property
    def arch_name(self) -> str:
        r"""Return the arch name of the main package.

        Returns
        -------
        arch_name : str
            The arch name of the main package, e.g. 'LEGATE_ARCH'.
        """
        return self._arch_name

    @property
    def arch_value(self) -> str:
        r"""Return the arch value of the main package.

        Returns
        -------
        arch_value : str
            The arch value of the main package, e.g. 'arch-darwin-debug'.
        """
        return self._arch_value

    @property
    def arch_value_provenance(self) -> ValueProvenance:
        r"""Get the provenance of the arch value.

        Returns
        -------
        provenance : ValueProvenance
            The provenance of the arch value.
        """
        return self._arch_value_provenance

    @property
    def project_dir_name(self) -> str:
        r"""Get the project dir name of the main package.

        Returns
        -------
        proj_dir_name : str
            The name of the project dir variable, e.g. 'LEGATE_DIR'.
        """
        return self._proj_dir_name

    @property
    def project_dir_value(self) -> Path:
        r"""Get the project dir value of the main package.

        Returns
        -------
        proj_dir_value : Path
            The value of the project dir variable,
            e.g. /path/to/legate.internal.
        """
        return self._proj_dir_value

    @property
    def project_src_dir(self) -> Path:
        r"""Get the source directory of the main package.

        Returns
        -------
        proj_dir_value : Path
            The project source dir e.g. /path/to/legate.internal/src.
        """
        return self._proj_src_dir

    @staticmethod
    def _preparse_value(
        argv: Sequence[str],
        opt_name: str,
        bool_opt: bool = False,
        environ_name: str | None = None,
    ) -> tuple[str | None, ValueProvenance]:
        r"""Parse out a value from command line and environment.

        Parameters
        ----------
        argv : Sequence[str]
            The command line to parse.
        opt_name : str
            The name of the command line option to extract.
        bool_opt : False
            True if `opt_name` refers to a boolean option, False otherwise.
        environ_name : str, optional
            The name of the environment variables to parse (if any).

        Raises
        ------
        AssertionError
            If `opt_name` does not start with a '-'.
        """
        assert opt_name.startswith(
            "-"
        ), f"Option name '{opt_name}' must start with '-'"
        dest_name = flag_to_dest(opt_name)
        parser = ArgumentParser(add_help=False)
        if bool_opt:
            parser.add_argument(
                opt_name,
                nargs="?",
                const=True,
                default=None,
                type=_str_to_bool,
                dest=dest_name,
            )
        else:
            parser.add_argument(opt_name, required=False, dest=dest_name)
        args, _ = parser.parse_known_args(argv)

        if (val := getattr(args, dest_name)) is not None:
            return val, ValueProvenance.COMMAND_LINE

        if environ_name is not None and (
            (val := os.environ.get(environ_name, None)) is not None
        ):
            return val, ValueProvenance.ENVIRONMENT
        return None, ValueProvenance.GENERATED  # not found

    def preparse_arch_value(
        self, argv: Sequence[str]
    ) -> tuple[str, ValueProvenance]:
        r"""Pre-parse (or generate) the project ARCH value based on argv.

        Parameters
        ----------
        argv : Sequence[str]
            The command-line arguments to search

        Returns
        -------
        arch : str
            The value of the found or generated ARCH
        provenance : ValueProvenance
            The provenance of the arch value, detailing where the value was
            found.
        """
        arch, provenance = self._preparse_value(
            argv, f"--{self.arch_name}", environ_name=self.arch_name
        )
        if arch is not None:
            # found something
            return arch, provenance

        gen_arch = ["arch", platform.system().casefold()]
        have_py, _ = self._preparse_value(argv, "--with-python", bool_opt=True)
        if have_py:
            gen_arch.append("py")
        have_cuda, _ = self._preparse_value(argv, "--with-cuda", bool_opt=True)
        if have_cuda:
            gen_arch.append("cuda")
        build_type, _ = self._preparse_value(
            argv,
            self.CMAKE_BUILD_TYPE.name,
            environ_name=str(self.CMAKE_BUILD_TYPE.cmake_var),
        )
        if build_type is None:
            build_type = _DEFAULT_BUILD_TYPE
        else:
            build_type = build_type.casefold()
        gen_arch.append(build_type)
        return "-".join(gen_arch), ValueProvenance.GENERATED

    def add_options(self, parser: ArgumentParser) -> None:
        r"""Add options for the main package.

        Parameters
        ----------
        parser : ArgumentParser
            The argument parser to add options to.
        """
        # first do the base options
        base_group = self.create_argument_group(parser, title="Base Options")
        base_group.add_argument(
            f"--{self.arch_name}",
            help=f"{self.manager.project_name} build directory",
            default=self.manager.project_arch,
        )
        self.log_execute_func(
            self.add_package_options, base_group, ignored_only=True
        )
        # then do the options for the derived main package
        package_group = self.create_argument_group(parser, title=self.name)
        self.log_execute_func(self.add_package_options, package_group)

    def inspect_packages(self, packages: Sequence[Package]) -> None:
        r"""Inspect the set of packages loaded byx the configuration manager.

        Parameters
        ----------
        packages : Sequence[Package]
            The packages to inspect.

        Notes
        -----
        This routine may be used to add, remove, or otherwise modify the
        loaded packages. For example, the main package may want to set certain
        packages to always be enabled, or perhaps would like to remove other
        packages (for which it provides no support).

        By default, no packages are removed.
        """
        pass

    def configure_core_package_variables(self) -> None:
        r"""Configure the core main package cmake variables."""
        self.manager.set_cmake_variable(
            self.CMAKE_EXPORT_COMPILE_COMMANDS, True
        )
        self.manager.set_cmake_variable(self.CMAKE_COLOR_DIAGNOSTICS, True)
        self.manager.set_cmake_variable(self.CMAKE_COLOR_MAKEFILE, True)
        self.manager.set_cmake_variable(
            self.CMAKE_BUILD_PARALLEL_LEVEL, self.cl_args.num_threads.value
        )
        match self.cl_args.library_linkage.value:
            case LibraryLinkage.SHARED:
                self.manager.set_cmake_variable(self.BUILD_SHARED_LIBS, True)
            case LibraryLinkage.STATIC:
                self.manager.set_cmake_variable(self.BUILD_SHARED_LIBS, False)
        self.set_flag_if_set(self.CMAKE_INSTALL_PREFIX, self.cl_args.prefix)

    def configure_c(self) -> None:
        r"""Configure C compiler variables."""
        self.set_flag_if_user_set(self.CMAKE_C_COMPILER, self.cl_args.CC)
        self._configure_language_flags(self.CMAKE_C_FLAGS, self.cl_args.CFLAGS)

    def configure_cxx(self) -> None:
        r"""Configure C++ compiler variables."""
        self.set_flag_if_user_set(self.CMAKE_CXX_COMPILER, self.cl_args.CXX)
        self._configure_language_flags(
            self.CMAKE_CXX_FLAGS, self.cl_args.CXXFLAGS
        )

    def setup(self) -> None:
        r"""Setup the Main Package."""
        # We do this here because the compilers need to know what the build
        # type is to set reasonable defaults in their configure(). Because we
        # cannot guarantee that the main package is configured first (in fact,
        # in almost all cases it isn't), we must do this here.
        self.manager.set_cmake_variable(
            self.CMAKE_BUILD_TYPE,
            _CMAKE_BUILD_TYPE_MAP[self.cl_args.build_type.value],
        )
        super().setup()

    def configure(self) -> None:
        r"""Configure the Main Package."""
        super().configure()
        self.log_execute_func(self.configure_core_package_variables)
        self.log_execute_func(self.configure_c)
        self.log_execute_func(self.configure_cxx)

    def finalize(self) -> None:
        r"""Finalize the Main package."""
        super().finalize()
        self.manager.add_gmake_search_variable(self.CMAKE_BUILD_PARALLEL_LEVEL)

    def finalize_default_arch_file(self) -> None:
        r"""Emit a file containing this configuration's PROJECT_ARCH so
        that the user doesn't have to have it defined in env.
        """
        path = self._default_arch_file_path
        if path is None:
            self.log("Default arch file path is None, not emitting file")
            return

        from datetime import date
        from stat import S_IEXEC

        path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.log(f"Emitting {self.arch_value!r} to {path}")
        text = DEFAULT_PROJECT_ARCH_SCRIPT_TEMPLATE.format(
            YEAR=date.today().year,
            FILE=__file__,
            PROJECT_NAME=self.project_name.casefold(),
            PROJECT_ARCH_VALUE=self.arch_value,
        )
        self.log(f"Emitting to {path}:\n\n{text}")
        path.write_text(text)
        path.chmod(path.stat().st_mode | S_IEXEC)

    def post_finalize(self) -> None:
        r"""Execute finalization for the main package only after
        successful configure run.
        """
        self.log_execute_func(self.finalize_default_arch_file)

    def summarize_main(self) -> str:
        r"""Provide the main summary for the Main Package.

        Returns
        -------
        summary : str
            The summary

        Notes
        -----
        This is different from `Package.summarize()`. It should be appended
        in addition to the former to the package summary.
        """
        ret = [
            self.create_package_summary(
                [
                    (
                        f"{self.manager.project_name} Dir",
                        self.manager.project_dir,
                    ),
                    (
                        f"{self.manager.project_name} Arch",
                        self.manager.project_arch,
                    ),
                    (
                        "Build Generator",
                        self.manager.read_cmake_variable("CMAKE_MAKE_PROGRAM"),
                    ),
                    (
                        "Build type",
                        self.manager.get_cmake_variable(self.CMAKE_BUILD_TYPE),
                    ),
                    (
                        "Num Build Threads",
                        self.manager.read_cmake_variable(
                            self.CMAKE_BUILD_PARALLEL_LEVEL
                        ),
                    ),
                    (
                        "Install prefix",
                        self.manager.read_cmake_variable(
                            self.CMAKE_INSTALL_PREFIX
                        ),
                    ),
                ],
                title="Core Project",
            )
        ]

        def summarize_compiler(
            name: str,
            cmake_compiler_var: ConfigArgument,
            compiler_attr_name: str,
            cmake_flags_var: ConfigArgument,
            flags_attr_name: str,
        ) -> str:
            try:
                cc = self.manager.read_cmake_variable(cmake_compiler_var)
            except ValueError:
                cc = getattr(self.cl_args, compiler_attr_name).value

            version = self.log_execute_command([cc, "--version"]).stdout

            ccflags: str | None | list[str] | tuple[str, ...]
            try:
                ccflags = self.manager.read_cmake_variable(cmake_flags_var)
            except ValueError:
                ccflags = getattr(self.cl_args, flags_attr_name).value
                if isinstance(ccflags, (list, tuple)):
                    ccflags = " ".join(ccflags)

            if not ccflags:
                ccflags = "[]"

            return self.create_package_summary(
                [
                    ("Executable", cc),
                    ("Version", version),
                    (f"Global {name} Flags", ccflags),
                ],
                title=f"{name} Compiler",
            )

        ret.append(
            summarize_compiler(
                "C", self.CMAKE_C_COMPILER, "CC", self.CMAKE_C_FLAGS, "CFLAGS"
            )
        )
        ret.append(
            summarize_compiler(
                "C++",
                self.CMAKE_CXX_COMPILER,
                "CXX",
                self.CMAKE_CXX_FLAGS,
                "CXXFLAGS",
            )
        )
        return "\n".join(ret)
