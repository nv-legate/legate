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

import re
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from ..aedifix import (
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

if TYPE_CHECKING:
    from ..aedifix.package.packages.cal import CAL
    from ..aedifix.package.packages.cmake import CMake
    from ..aedifix.package.packages.legion import Legion
    from ..aedifix.package.packages.python import Python


class Legate(MainPackage):
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
    legate_BUILD_EXAMPLES: Final = ConfigArgument(
        name="--with-examples",
        spec=ArgSpec(dest="with_examples", type=bool, help="Build examples"),
        cmake_var=CMAKE_VARIABLE("legate_BUILD_EXAMPLES", CMakeBool),
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
            dest="legate_cxx_flags",
            nargs=1,
            help="C++ flags for Legate",
        ),
        cmake_var=CMAKE_VARIABLE("legate_CXX_FLAGS", CMakeList),
    )
    legate_CUDA_FLAGS: Final = ConfigArgument(
        name="--legate-cuda-flags",
        spec=ArgSpec(
            dest="legate_cuda_flags",
            nargs=1,
            help="CUDA flags for Legate",
        ),
        cmake_var=CMAKE_VARIABLE("legate_CUDA_FLAGS", CMakeList),
    )
    legate_LINKER_FLAGS: Final = ConfigArgument(
        name="--legate-linker-flags",
        spec=ArgSpec(
            dest="legate_linker_flags",
            nargs=1,
            help="Linker flags for Legate",
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
    legate_USE_CAL: Final = CMAKE_VARIABLE("legate_USE_CAL", CMakeBool)

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
        from scripts.get_legate_dir import get_legate_dir

        legate_dir = Path(get_legate_dir())
        super().__init__(
            manager=manager,
            argv=argv,
            name="Legate",
            arch_name="LEGATE_ARCH",
            project_dir_name="LEGATE_DIR",
            project_dir_value=legate_dir,
            project_config_file_template=(
                Path(__file__).parent / "gmakevariables.in"
            ),
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

    def declare_dependencies(self) -> None:
        r"""Declare Dependencies for Legate."""
        super().declare_dependencies()
        self.cmake: CMake = self.require("cmake")  # type: ignore[assignment]
        self.legion: Legion = self.require(  # type: ignore[assignment]
            "legion"
        )
        self.python: Python = self.require(  # type: ignore[assignment]
            "python"
        )
        self.cal: CAL = self.require("cal")  # type: ignore[assignment]

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
            import pip  # noqa: F401
        except ModuleNotFoundError as mnfe:
            self.log(
                f"pip does not appear to be installed: '{mnfe}'. Nothing to do"
            )
            return

        installed_packages = self.log_execute_command(
            [sys.executable, "-m", "pip", "list"]
        ).stdout.splitlines()
        # skip the "Package Version" header and divider lines
        installed_packages = installed_packages[2:]
        package_names = (
            line.split(maxsplit=1)[0] for line in installed_packages
        )
        found_legate = any(name.startswith("legate") for name in package_names)
        self.log(f"Have pre-existing legate installation: {found_legate}")

        if not found_legate:
            return

        if self.cl_args.with_clean.value:
            cmd = [
                sys.executable,
                "-m",
                "pip",
                "uninstall",
                "--yes",
                "legate",
            ]
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
        r"""Setup Legate"""
        self.log_execute_func(self.maybe_uninstall_legate)
        super().setup()

    def check_min_cmake_version(self) -> None:
        r"""Assert the minimum cmake version is met."""
        try:
            from packaging.version import parse as version_parse
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
                raise RuntimeError(
                    "Failed to parse minimum required CMake version from"
                    f" {cmakelists_txt}"
                )

        self.log(f"Minimum cmake version required: {min_ver}")
        if version_parse(self.cmake.version) < version_parse(min_ver):
            cmake_exe = self.manager.get_cmake_variable(
                self.cmake.CMAKE_COMMAND
            )
            raise RuntimeError(
                f"CMake executable {cmake_exe} too old! Expected version "
                f"{min_ver}, have {self.cmake.version}"
            )

    def configure_legate_variables(self) -> None:
        r"""Configure the general variables for Legate."""
        self.append_flags_if_set(
            self.legate_CXX_FLAGS, self.cl_args.legate_cxx_flags
        )
        self.append_flags_if_set(
            self.legate_LINKER_FLAGS,
            self.cl_args.legate_linker_flags,
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
            self.legate_BUILD_EXAMPLES, self.cl_args.with_examples
        )
        self.set_flag_if_user_set(
            self.legate_BUILD_BENCHMARKS, self.cl_args.with_benchmarks
        )
        self.set_flag_if_user_set(self.BUILD_MARCH, self.cl_args.build_march)
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
                f" 1. ./configure --{flag_name}=0 --with-foo (CMake downloads and builds libfoo.so)\n"  # noqa e501
                f" 2. pip install . (CMake -- as a byproduct of installing {self.project_name} -- installs libfoo.so)\n"  # noqa E501
                " 3. ./reconfigure... (CMake now picks up installed libfoo.so instead of reusing the downloaded one)\n"  # noqa E501
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

    def configure_legion(self) -> None:
        r"""Configure Legion for use with Legate."""
        self.set_flag_if_user_set(
            self.legate_LEGION_BRANCH, self.cl_args.legion_branch
        )
        if self.python.state.enabled():
            self.manager.set_cmake_variable(
                self.legion.Legion_BUILD_BINDINGS, False
            )

    def configure_clang_tidy(self) -> None:
        r"""Configure clang-tidy variables."""
        self.set_flag_if_user_set(
            self.LEGATE_CLANG_TIDY, self.cl_args.clang_tidy_executable
        )

    def configure_cal(self) -> None:
        r"""Configure CAL variables."""
        cal_state = self.cal.state
        if cal_state.enabled():
            self.manager.set_cmake_variable(self.legate_USE_CAL, True)
        elif cal_state.explicitly_disabled():
            self.manager.set_cmake_variable(self.legate_USE_CAL, False)

    def configure(self) -> None:
        r"""Configure Legate."""
        super().configure()
        self.log_execute_func(self.check_min_cmake_version)
        self.log_execute_func(self.configure_legate_variables)
        self.log_execute_func(self.configure_legion)
        self.log_execute_func(self.configure_clang_tidy)
        self.log_execute_func(self.configure_cal)

    def _summarize_flags(self) -> list[tuple[str, Any]]:
        def make_summary(
            name: str, cmake_varname: ConfigArgument
        ) -> tuple[str, str]:
            flags = self.manager.get_cmake_variable(cmake_varname)
            if not flags:
                flags = self.manager.read_cmake_variable(cmake_varname)
            if isinstance(flags, (list, tuple)):
                flags = " ".join(flags)
            return (f"{name} Flags", flags.replace(";", " "))

        return [
            make_summary("C++", self.legate_CXX_FLAGS),
            make_summary("Linker", self.legate_LINKER_FLAGS),
            make_summary("CUDA", self.legate_CUDA_FLAGS),
        ]

    def _summarize_python(self) -> list[tuple[str, Any]]:
        python = self.python
        py_enabled = python.state.enabled()
        lines: list[tuple[str, Any]] = [("Python bindings", py_enabled)]
        if py_enabled:
            try:
                lines.append(("Python library path", python.lib_path))
            except AttributeError:
                # no python.lib_path
                pass
            try:
                lines.append(("Python library version", python.lib_version))
            except AttributeError:
                # no python.lib_version
                pass
        return lines

    def _summarize_misc(self) -> list[tuple[str, Any]]:
        return [
            (
                "With Tests",
                bool(self.manager.get_cmake_variable(self.legate_BUILD_TESTS)),
            ),
            (
                "With Docs",
                bool(self.manager.get_cmake_variable(self.legate_BUILD_DOCS)),
            ),
            (
                "With Examples",
                bool(
                    self.manager.get_cmake_variable(self.legate_BUILD_EXAMPLES)
                ),
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
