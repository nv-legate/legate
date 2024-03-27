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

import re
import shutil
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
    Package,
)

if TYPE_CHECKING:
    from ..aedifix.package.packages.cmake import CMake
    from ..aedifix.package.packages.legion import Legion
    from ..aedifix.package.packages.python import Python


class LegateCore(MainPackage):
    legate_core_BUILD_DOCS: Final = ConfigArgument(
        name="--with-docs",
        spec=ArgSpec(
            dest="with_docs", type=bool, help="Build Doxygen documentation"
        ),
        cmake_var=CMAKE_VARIABLE("legate_core_BUILD_DOCS", CMakeBool),
    )
    legate_core_BUILD_TESTS: Final = ConfigArgument(
        name="--with-tests",
        spec=ArgSpec(dest="with_tests", type=bool, help="Build tests"),
        cmake_var=CMAKE_VARIABLE("legate_core_BUILD_TESTS", CMakeBool),
    )
    legate_core_BUILD_EXAMPLES: Final = ConfigArgument(
        name="--with-examples",
        spec=ArgSpec(dest="with_examples", type=bool, help="Build examples"),
        cmake_var=CMAKE_VARIABLE("legate_core_BUILD_EXAMPLES", CMakeBool),
    )
    legate_core_CXX_FLAGS: Final = ConfigArgument(
        name="--legate-core-cxx-flags",
        spec=ArgSpec(
            dest="legate_core_cxx_flags",
            nargs=1,
            help="C++ flags for Legate.Core",
        ),
        cmake_var=CMAKE_VARIABLE("legate_core_CXX_FLAGS", CMakeList),
    )
    legate_core_CUDA_FLAGS: Final = ConfigArgument(
        name="--legate-core-cuda-flags",
        spec=ArgSpec(
            dest="legate_core_cuda_flags",
            nargs=1,
            help="CUDA flags for Legate.Core",
        ),
        cmake_var=CMAKE_VARIABLE("legate_core_CUDA_FLAGS", CMakeList),
    )
    legate_core_LINKER_FLAGS: Final = ConfigArgument(
        name="--legate-core-linker-flags",
        spec=ArgSpec(
            dest="legate_core_linker_flags",
            nargs=1,
            help="Linker flags for Legate.Core",
        ),
        cmake_var=CMAKE_VARIABLE("legate_core_LINKER_FLAGS", CMakeList),
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
    LEGATE_CORE_CLANG_TIDY: Final = ConfigArgument(
        name="--clang-tidy-executable",
        spec=ArgSpec(
            dest="clang_tidy_executable",
            type=Path,
            default=shutil.which("clang-tidy"),
            help="clang-tidy executable",
        ),
        cmake_var=CMAKE_VARIABLE("LEGATE_CORE_CLANG_TIDY", CMakeExecutable),
    )
    LEGATE_CORE_RUN_CLANG_TIDY: Final = ConfigArgument(
        name="--run-clang-tidy-executable",
        spec=ArgSpec(
            dest="run_clang_tidy_executable",
            type=Path,
            default=shutil.which("run-clang-tidy"),
            help="run-clang-tidy executable",
        ),
        cmake_var=CMAKE_VARIABLE(
            "LEGATE_CORE_RUN_CLANG_TIDY", CMakeExecutable
        ),
    )
    legate_core_LEGION_REPOSITORY: Final = CMAKE_VARIABLE(
        "legate_core_LEGION_REPOSITORY", CMakeString
    )
    legate_core_LEGION_BRANCH: Final = CMAKE_VARIABLE(
        "legate_core_LEGION_BRANCH", CMakeString
    )

    def __init__(
        self, manager: ConfigurationManager, argv: Sequence[str]
    ) -> None:
        r"""Construct a Legate.Core main package.

        Parameters
        ----------
        manager : ConfigurationManager
            The configuration manager to manage this package.
        argv : Sequence[str]
            The command line arguments for this configuration.
        """
        super().__init__(
            manager=manager,
            argv=argv,
            name="Legate.Core",
            arch_name="LEGATE_CORE_ARCH",
            project_dir_name="LEGATE_CORE_DIR",
            project_dir_value=Path(__file__).resolve().parent.parent.parent,
        )

    @classmethod
    def from_argv(
        cls, manager: ConfigurationManager, argv: Sequence[str]
    ) -> LegateCore:
        r"""Construct a Legate.Core main package from argv.

        Parameters
        ----------
        manager : ConfigurationManager
            The configuration manager to manage this package.
        argv : Sequence[str]
            The command line arguments for this configuration

        Returns
        -------
        package : LegateCore
            The constructed main package.
        """
        return cls(manager, argv)

    def inspect_packages(self, packages: Sequence[Package]) -> None:
        r"""Inspect the set of packages loaded by the configuration manager.

        Parameters
        ----------
        packages : Sequence[Package]
            The packages to inspect.
        """
        always_enabled = {"legion": 0, "thrust": 0, "cmake": 0}
        for pack in packages:
            pack_name = pack.name
            try:
                always_enabled[pack_name.casefold()] += 1
            except KeyError:
                continue
            self.log(f"Force-enabling '{pack_name}'")

        for name, init_cnt in always_enabled.items():
            if not init_cnt:
                raise AssertionError(
                    f"Did not find package '{name}' during package inspection,"
                    " has it been removed?"
                )
            if init_cnt > 1:
                raise AssertionError(
                    f"Duplicate packages for '{name}'? Initialized it "
                    f"{init_cnt} times"
                )

    def declare_dependencies(self) -> None:
        r"""Declare Dependencies for Legate.Core."""
        super().declare_dependencies()
        self.cmake: CMake = self.require("cmake")  # type: ignore[assignment]
        self.legion: Legion = self.require(  # type: ignore[assignment]
            "legion"
        )
        self.python: Python = self.require(  # type: ignore[assignment]
            "python"
        )

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
                return tuple(map(int, version.split(".")))

        min_ver_re = re.compile(
            r"cmake_minimum_required\(.*VERSION\s+([^\s]+)"
        )
        cmakelists_txt = self.project_dir / "CMakeLists.txt"
        for line in cmakelists_txt.open():
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
                self.cmake.CMAKE_EXECUTABLE
            )
            raise RuntimeError(
                f"CMake executable {cmake_exe} too old! Expected version "
                f"{min_ver}, have {self.cmake.version}"
            )

    def configure_core_variables(self) -> None:
        r"""Configure the general variables for Legate.Core."""
        self.append_flags_if_set(
            self.legate_core_CXX_FLAGS, self.cl_args.legate_core_cxx_flags
        )
        self.append_flags_if_set(
            self.legate_core_LINKER_FLAGS,
            self.cl_args.legate_core_linker_flags,
        )
        self.append_flags_if_set(
            self.legate_core_CUDA_FLAGS, self.cl_args.legate_core_cuda_flags
        )
        self.set_flag_if_user_set(
            self.legate_core_BUILD_DOCS, self.cl_args.with_docs
        )
        self.set_flag_if_user_set(
            self.legate_core_BUILD_TESTS, self.cl_args.with_tests
        )
        self.set_flag_if_user_set(
            self.legate_core_BUILD_EXAMPLES, self.cl_args.with_examples
        )
        self.set_flag_if_user_set(self.BUILD_MARCH, self.cl_args.build_march)

    def configure_legion(self) -> None:
        r"""Configure Legion for use with Legate.Core."""
        self.set_flag_if_user_set(
            self.legate_core_LEGION_REPOSITORY, self.cl_args.legion_url
        )
        self.set_flag_if_user_set(
            self.legate_core_LEGION_BRANCH, self.cl_args.legion_branch
        )

    def configure_clang_tidy(self) -> None:
        r"""Configure clang-tidy variables."""
        self.set_flag_if_user_set(
            self.LEGATE_CORE_CLANG_TIDY, self.cl_args.clang_tidy_executable
        )
        self.set_flag_if_user_set(
            self.LEGATE_CORE_RUN_CLANG_TIDY,
            self.cl_args.run_clang_tidy_executable,
        )

    def configure(self) -> None:
        r"""Configure Legate.Core."""
        super().configure()
        self.log_execute_func(self.check_min_cmake_version)
        self.log_execute_func(self.configure_core_variables)
        self.log_execute_func(self.configure_legion)
        self.log_execute_func(self.configure_clang_tidy)

    def finalize(self) -> None:
        r"""Finalize Legate.Core."""
        self.manager.add_gmake_variable(
            "LEGATE_CORE_USE_PYTHON",
            "1" if self.python.state.value else "0",
            override_ok=False,
        )

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
            make_summary("C++", self.legate_core_CXX_FLAGS),
            make_summary("Linker", self.legate_core_LINKER_FLAGS),
            make_summary("CUDA", self.legate_core_CUDA_FLAGS),
        ]

    def _summarize_python(self) -> list[tuple[str, Any]]:
        python = self.python
        py_enabled = python.state.value
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
                bool(
                    self.manager.get_cmake_variable(
                        self.legate_core_BUILD_TESTS
                    )
                ),
            ),
            (
                "With Docs",
                bool(
                    self.manager.get_cmake_variable(
                        self.legate_core_BUILD_DOCS
                    )
                ),
            ),
            (
                "With Examples",
                bool(
                    self.manager.get_cmake_variable(
                        self.legate_core_BUILD_EXAMPLES
                    )
                ),
            ),
        ]

    def summarize(self) -> str:
        r"""Summarize Legate.Core.

        Returns
        -------
        summary : str
            The summary of Legate.Core.
        """
        lines = []
        for summarizer in (
            self._summarize_flags,
            self._summarize_misc,
            self._summarize_python,
        ):
            lines.extend(summarizer())
        return self.create_package_summary(lines)
