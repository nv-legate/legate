#!/usr/bin/env python3
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
import re
import sys
import textwrap
from copy import deepcopy
from pathlib import Path

import pytest

from ..manager import ConfigurationManager
from ..package.main_package import DEBUG_CONFIGURE_FLAG, ON_ERROR_DEBUGGER_FLAG
from ..util.cl_arg import CLArg
from ..util.exception import WrongOrderError
from .fixtures.dummy_main_module import DummyMainModule


@pytest.fixture
def manager() -> ConfigurationManager:
    return ConfigurationManager(tuple(), DummyMainModule)


class TestConfigurationManager:
    @pytest.mark.parametrize(
        "argv", (tuple(), ("--foo",), ("-b", "1", "--bar=baz"))
    )
    def test_create(
        self,
        argv: tuple[str, ...],
        AEDIFIX_PYTEST_DIR: Path,
        AEDIFIX_PYTEST_ARCH: str,
    ) -> None:
        manager = ConfigurationManager(argv, DummyMainModule)
        assert os.environ["AEDIFIX"] == "1"
        assert manager.argv == argv
        with pytest.raises(
            WrongOrderError, match=re.escape("Must call setup() first")
        ):
            manager.cl_args
        assert manager.project_name == "DummyMainModule"
        assert manager.project_arch == AEDIFIX_PYTEST_ARCH
        assert manager.project_arch_name == "AEDIFIX_PYTEST_ARCH"
        assert manager.project_dir == AEDIFIX_PYTEST_DIR
        # This dir is created by the pytest fixtures, but let's just double
        # check that it still exists
        assert manager.project_dir.exists()
        assert manager.project_dir.is_dir()
        assert manager.project_dir_name == "AEDIFIX_PYTEST_DIR"
        assert (
            manager.project_arch_dir
            == AEDIFIX_PYTEST_DIR / AEDIFIX_PYTEST_ARCH
        )
        assert not manager.project_arch_dir.exists()
        assert (
            manager.project_cmake_dir
            == AEDIFIX_PYTEST_DIR / AEDIFIX_PYTEST_ARCH / "cmake_build"
        )
        assert not manager.project_cmake_dir.exists()

        # This should not exist yet, because the manager should not have
        # emitted any logging yet!
        assert not manager._logger.file_path.exists()

        assert manager._aedifix_root_dir.exists()
        assert manager._aedifix_root_dir.is_dir()
        assert (manager._aedifix_root_dir / "aedifix").exists()
        assert (manager._aedifix_root_dir / "aedifix").is_dir()

    def test_setup(
        self, manager: ConfigurationManager, AEDIFIX_PYTEST_ARCH: str
    ) -> None:
        orig_argv = deepcopy(manager.argv)
        assert len(manager._modules) == 1
        manager.setup()
        assert len(manager._modules) > 1
        assert manager.argv == orig_argv
        assert manager.cl_args.AEDIFIX_PYTEST_ARCH == CLArg(
            name="AEDIFIX_PYTEST_ARCH", value=AEDIFIX_PYTEST_ARCH, cl_set=False
        )
        assert manager._ephemeral_args == {
            "--with-clean",
            ON_ERROR_DEBUGGER_FLAG,
            DEBUG_CONFIGURE_FLAG,
        }
        assert manager.project_dir.exists()
        assert manager.project_dir.is_dir()
        assert manager.project_arch_dir.exists()
        assert manager.project_arch_dir.is_dir()
        assert manager._logger.file_path.exists()
        assert manager._logger.file_path.is_file()

    @pytest.mark.slow
    def test_read_cmake_variable(
        self, manager: ConfigurationManager, AEDIFIX_PYTEST_DIR: Path
    ) -> None:
        main_cpp_template = textwrap.dedent(
            r"""
        #include <iostream>

        int main(int argc, char *argv[])
        {
          std::cout << "hello, world!\n";
          return 0;
        }
        """
        ).strip()
        cmakelists_template = textwrap.dedent(
            """
        cmake_minimum_required(VERSION 3.13...3.16 FATAL_ERROR)

        project(example_exec VERSION 0.0.1 LANGUAGES CXX)

        add_executable(example_exec src/main.cpp)

        set(
          MY_VARIABLE
          -foo=bar -baz=bop -hello --world
          CACHE STRING "My variable" FORCE
        )
        install(TARGETS example_exec)
        """
        ).strip()
        src_dir = AEDIFIX_PYTEST_DIR / "src"
        src_dir.mkdir()
        (src_dir / "main.cpp").write_text(main_cpp_template)
        (AEDIFIX_PYTEST_DIR / "CMakeLists.txt").write_text(cmakelists_template)
        manager.setup()
        manager.configure()
        manager.finalize()
        var = manager.read_cmake_variable("MY_VARIABLE")
        assert isinstance(var, str)
        assert var.split(";") == ["-foo=bar", "-baz=bop", "-hello", "--world"]

    @pytest.mark.slow
    def test_manager_extra_args(self, AEDIFIX_PYTEST_DIR: Path) -> None:
        main_cpp_template = textwrap.dedent(
            r"""
        #include <iostream>

        int main(int argc, char *argv[])
        {
          std::cout << "hello, world!\n";
          return 0;
        }
        """
        ).strip()
        cmakelists_template = textwrap.dedent(
            r"""
        cmake_minimum_required(VERSION 3.13...3.16 FATAL_ERROR)

        if(NOT DEFINED MY_VARIABLE)
          message(
            FATAL_ERROR
            "ConfigurationManager failed to forward extra arguments to CMake!"
          )
        endif()

        if(NOT ("${MY_VARIABLE}" STREQUAL "foo-bar-baz"))
          message(
            FATAL_ERROR
            "ConfigurationManager failed to forward extra arguments to CMake!"
          )
        endif()

        project(example_exec VERSION 0.0.1 LANGUAGES CXX)

        add_executable(example_exec src/main.cpp)

        install(TARGETS example_exec)
        """
        ).strip()
        src_dir = AEDIFIX_PYTEST_DIR / "src"
        src_dir.mkdir()
        (src_dir / "main.cpp").write_text(main_cpp_template)
        (AEDIFIX_PYTEST_DIR / "CMakeLists.txt").write_text(cmakelists_template)
        manager = ConfigurationManager(
            ("--", "-DMY_VARIABLE='foo-bar-baz'"), DummyMainModule
        )
        manager.setup()
        manager.configure()
        manager.finalize()
        var = manager.read_cmake_variable("MY_VARIABLE")
        assert isinstance(var, str)
        assert var == "foo-bar-baz"


if __name__ == "__main__":
    sys.exit(pytest.main())
