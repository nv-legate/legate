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

import json
import shutil
import sys
import textwrap
from pathlib import Path
from typing import Any, TypeAlias

import pytest
from pytest import CaptureFixture, MonkeyPatch

from ..logger import Logger
from ..main import basic_configure
from ..manager import ConfigurationManager
from ..util.constants import Constants
from .fixtures.dummy_main_module import DummyMainModule


@pytest.fixture(scope="function", autouse=True)
def setup_cmake_project(AEDIFIX_PYTEST_DIR: Path) -> None:
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
        target_compile_features(example_exec PRIVATE cxx_auto_type)

        install(TARGETS example_exec)
        """
    ).strip()
    src_dir = AEDIFIX_PYTEST_DIR / "src"
    src_dir.mkdir()
    (src_dir / "main.cpp").write_text(main_cpp_template)
    (AEDIFIX_PYTEST_DIR / "CMakeLists.txt").write_text(cmakelists_template)


def shutil_which(thing: str) -> Path:
    ret = shutil.which(thing)
    assert ret is not None
    return Path(ret)


Argv: TypeAlias = list[str]
CMakeSpec: TypeAlias = dict[str, str | list[str]]


class TestInfo:
    # tell pytest to ignore this class, even though it starts with "Test"
    __test__ = False

    def __init__(
        self,
        AEDIFIX_PYTEST_DIR: Path,
        AEDIFIX_PYTEST_ARCH: str,
        generator: str | None = None,
    ) -> None:
        self.AEDIFIX_PYTEST_DIR = AEDIFIX_PYTEST_DIR
        self.AEDIFIX_PYTEST_ARCH = AEDIFIX_PYTEST_ARCH
        self.arch_dir = self.AEDIFIX_PYTEST_DIR / self.AEDIFIX_PYTEST_ARCH
        self.configure_log = self.AEDIFIX_PYTEST_DIR / "configure.log"
        self.backup_configure_log = self.arch_dir / "configure.log"
        self.reconfigure = (
            self.arch_dir / f"reconfigure-{self.AEDIFIX_PYTEST_ARCH}.py"
        )
        self.reconfigure_symlink = (
            self.AEDIFIX_PYTEST_DIR / self.reconfigure.name
        )
        self.cmake_dir = self.arch_dir / "cmake_build"
        self.cmakecache_txt = self.cmake_dir / "CMakeCache.txt"
        self.command_spec = self.cmake_dir / "aedifix_cmake_command_spec.json"
        self.cmake_exe = Path(shutil_which("cmake")).resolve()
        if generator is None:
            generator = "Ninja" if shutil.which("ninja") else "Unix Makefiles"
        self.generator = generator

    def pre_test(self) -> None:
        assert not self.arch_dir.exists()
        assert not self.configure_log.exists()
        assert not self.backup_configure_log.exists()
        assert not self.reconfigure.exists()
        assert not self.reconfigure_symlink.exists()
        assert not self.cmake_dir.exists()
        assert not self.cmakecache_txt.exists()
        assert not self.command_spec.exists()

    def post_test(self, argv: Argv, expected_spec: CMakeSpec) -> None:
        # basics
        assert self.arch_dir.is_dir()
        # configure.log
        assert self.configure_log.exists()
        assert self.configure_log.is_file()

        assert self.backup_configure_log.exists()
        assert self.backup_configure_log.is_file()
        assert (
            self.configure_log.read_text()
            == self.backup_configure_log.read_text()
        )

        # reconfigure
        assert self.reconfigure.exists()
        assert self.reconfigure.is_file()
        argv_lines = []
        with self.reconfigure.open() as fd:
            capturing = False
            for line in map(str.strip, fd):
                if line.startswith("argv = ["):
                    capturing = True
                elif capturing:
                    if line.startswith("] + sys.argv[1:]"):
                        break
                    argv_lines.append(line)
        # argv_lines should alwyas be one longer than argv, since configure
        # will insert the implicit --AEDIFIX_PYTEST_ARCH=<whatever> as the
        # first argument
        expected_argv = [
            f'"--AEDIFIX_PYTEST_ARCH={self.AEDIFIX_PYTEST_ARCH}",'
        ] + [f'"{arg}",' for arg in argv]
        assert argv_lines == expected_argv

        assert self.reconfigure_symlink.exists()
        assert self.reconfigure_symlink.is_symlink()
        assert (
            self.AEDIFIX_PYTEST_DIR / self.reconfigure_symlink.readlink()
            == self.reconfigure
        )

        # cmake dir
        assert self.cmake_dir.exists()
        assert self.cmake_dir.is_dir()

        # TODO: check more cmake cache!
        assert self.cmakecache_txt.exists()
        assert self.cmakecache_txt.is_file()
        cache_header_lines = [
            "# This is the CMakeCache file.\n",
            f"# For build in directory: {self.cmake_dir}\n",
            f"# It was generated by CMake: {self.cmake_exe}\n",
            "# You can edit this file to change values found and used by cmake.\n",  # noqa: E501
            "# If you do not want to change any of the values, simply exit the editor.\n",  # noqa: E501
            "# If you do want to change a value, simply edit, save, and exit the editor.\n",  # noqa: E501
            "# The syntax for the file is as follows:\n",
            "# KEY:TYPE=VALUE\n",
            "# KEY is the name of a variable in the cache.\n",
            "# TYPE is a hint to GUIs for the type of VALUE, DO NOT EDIT TYPE!.\n",  # noqa: E501
            "# VALUE is the current value for the KEY.\n",
        ]
        with self.cmakecache_txt.open() as fd:
            min_lines = len(cache_header_lines)
            # Exploit the fact that zip() will end when the shortest iterator
            # is exhausted (i.e. cache_header_lines in this case)
            for idx, (line, expected) in enumerate(
                zip(fd, cache_header_lines)
            ):
                assert line == expected
            # But double check the fact that cache_header_lines was indeed the
            # shortest
            assert idx + 1 == min_lines

        assert self.command_spec.exists()
        assert self.command_spec.is_file()
        with self.command_spec.open() as fd:
            spec = json.load(fd)
        assert spec == expected_spec


class SpecialException(Exception):
    pass


@pytest.mark.slow
class TestMain:
    def test_basic_configure_bad_init(self, monkeypatch: MonkeyPatch) -> None:
        exn_mess = "Throwing from __init__"

        def throwing_init(*args: Any, **kwargs: Any) -> None:
            raise SpecialException(exn_mess)

        monkeypatch.setattr(ConfigurationManager, "__init__", throwing_init)
        with pytest.raises(SpecialException, match=exn_mess):
            basic_configure(tuple(), DummyMainModule)
            # We should not get here, since if ConfigurationManager fails
            # to construct, then basic_configure() can do nothing but allow
            # the exception to percolate up.
            pytest.fail("Should not get here")

    def test_basic_configure_bad_halfway(
        self,
        monkeypatch: MonkeyPatch,
        capsys: CaptureFixture[str],
        AEDIFIX_PYTEST_DIR: Path,
    ) -> None:
        exn_mess = "Throwing from setup"

        def throwing_setup(*args: Any, **kwargs: Any) -> None:
            raise SpecialException(exn_mess)

        monkeypatch.setattr(ConfigurationManager, "setup", throwing_setup)

        ret = basic_configure(tuple(), DummyMainModule)
        assert ret != 0

        configure_log = AEDIFIX_PYTEST_DIR / "configure.log"
        assert configure_log.exists()
        assert configure_log.is_file()
        assert len(configure_log.read_text().strip())

        lines = capsys.readouterr().out.splitlines()
        expected_lines = Logger.build_multiline_message(
            sup_title="CONFIGURATION CRASH",
            divider_char="-",
            text=(
                f"{exn_mess}, please see {configure_log} for additional "
                "details."
            ),
        ).splitlines()
        banner = "=" * Constants.banner_length
        expected_lines.insert(0, banner)
        expected_lines.append(banner)
        assert lines == expected_lines

    def test_basic_configure_bare(
        self, AEDIFIX_PYTEST_DIR: Path, AEDIFIX_PYTEST_ARCH: str
    ) -> None:
        test_info = TestInfo(AEDIFIX_PYTEST_DIR, AEDIFIX_PYTEST_ARCH)
        test_info.pre_test()

        argv: Argv = []
        expected_spec: CMakeSpec = {
            "CMAKE_EXECUTABLE": f"{test_info.cmake_exe}",
            "CMAKE_GENERATOR": test_info.generator,
            "SOURCE_DIR": f"{AEDIFIX_PYTEST_DIR}",
            "BUILD_DIR": f"{test_info.cmake_dir}",
            "CMAKE_COMMANDS": [
                "--log-context",
                "--log-level=DEBUG",
                f"-DAEDIFIX_PYTEST_ARCH:STRING='{AEDIFIX_PYTEST_ARCH}'",
                f"-DAEDIFIX_PYTEST_DIR:PATH='{AEDIFIX_PYTEST_DIR}'",
                "-DBUILD_SHARED_LIBS:BOOL=ON",
                "-DCMAKE_BUILD_TYPE:STRING=Release",
                "-DCMAKE_COLOR_DIAGNOSTICS:BOOL=ON",
                "-DCMAKE_COLOR_MAKEFILE:BOOL=ON",
                "-DCMAKE_CXX_FLAGS:STRING=-O3",
                "-DCMAKE_C_FLAGS:STRING=-O3",
                "-DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=ON",
            ],
        }

        ret = basic_configure(argv, DummyMainModule)
        assert ret == 0
        test_info.post_test(argv, expected_spec)

    def test_basic_configure_release(
        self, AEDIFIX_PYTEST_DIR: Path, AEDIFIX_PYTEST_ARCH: str
    ) -> None:
        test_info = TestInfo(AEDIFIX_PYTEST_DIR, AEDIFIX_PYTEST_ARCH)
        test_info.pre_test()

        argv: Argv = ["--build-type=release"]
        expected_spec: CMakeSpec = {
            "CMAKE_EXECUTABLE": f"{test_info.cmake_exe}",
            "CMAKE_GENERATOR": test_info.generator,
            "SOURCE_DIR": f"{AEDIFIX_PYTEST_DIR}",
            "BUILD_DIR": f"{test_info.cmake_dir}",
            "CMAKE_COMMANDS": [
                "--log-context",
                "--log-level=DEBUG",
                f"-DAEDIFIX_PYTEST_ARCH:STRING='{AEDIFIX_PYTEST_ARCH}'",
                f"-DAEDIFIX_PYTEST_DIR:PATH='{AEDIFIX_PYTEST_DIR}'",
                "-DBUILD_SHARED_LIBS:BOOL=ON",
                "-DCMAKE_BUILD_TYPE:STRING=Release",
                "-DCMAKE_COLOR_DIAGNOSTICS:BOOL=ON",
                "-DCMAKE_COLOR_MAKEFILE:BOOL=ON",
                "-DCMAKE_CXX_FLAGS:STRING=-O3",
                "-DCMAKE_C_FLAGS:STRING=-O3",
                "-DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=ON",
            ],
        }

        ret = basic_configure(argv, DummyMainModule)
        assert ret == 0
        test_info.post_test(argv, expected_spec)

    def test_basic_configure_relwithdebinfo(
        self, AEDIFIX_PYTEST_DIR: Path, AEDIFIX_PYTEST_ARCH: str
    ) -> None:
        test_info = TestInfo(AEDIFIX_PYTEST_DIR, AEDIFIX_PYTEST_ARCH)
        test_info.pre_test()

        argv: Argv = ["--build-type=relwithdebinfo"]
        expected_spec: CMakeSpec = {
            "CMAKE_EXECUTABLE": f"{test_info.cmake_exe}",
            "CMAKE_GENERATOR": test_info.generator,
            "SOURCE_DIR": f"{AEDIFIX_PYTEST_DIR}",
            "BUILD_DIR": f"{test_info.cmake_dir}",
            "CMAKE_COMMANDS": [
                "--log-context",
                "--log-level=DEBUG",
                f"-DAEDIFIX_PYTEST_ARCH:STRING='{AEDIFIX_PYTEST_ARCH}'",
                f"-DAEDIFIX_PYTEST_DIR:PATH='{AEDIFIX_PYTEST_DIR}'",
                "-DBUILD_SHARED_LIBS:BOOL=ON",
                "-DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo",
                "-DCMAKE_COLOR_DIAGNOSTICS:BOOL=ON",
                "-DCMAKE_COLOR_MAKEFILE:BOOL=ON",
                "-DCMAKE_CXX_FLAGS:STRING='-O0 -g -g3 -O3'",
                "-DCMAKE_C_FLAGS:STRING='-O0 -g -g3 -O3'",
                "-DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=ON",
            ],
        }

        ret = basic_configure(argv, DummyMainModule)
        assert ret == 0
        test_info.post_test(argv, expected_spec)

    def test_basic_configure_clang_debug(
        self, AEDIFIX_PYTEST_DIR: Path, AEDIFIX_PYTEST_ARCH: str
    ) -> None:
        test_info = TestInfo(
            AEDIFIX_PYTEST_DIR, AEDIFIX_PYTEST_ARCH, generator="Unix Makefiles"
        )
        test_info.pre_test()

        flags = " ".join(
            [
                "-O0",
                "-g3",
                "-fstack-protector",
                "-Walloca",
                "-Wdeprecated",
                "-Wimplicit-fallthrough",
                "-fdiagnostics-show-template-tree",
                "-Wignored-qualifiers",
                "-Wmissing-field-initializers",
                "-Wextra",
                "-fsanitize=address,undefined,bounds",
            ]
        )

        cc = Path(shutil_which("clang"))
        cxx = Path(shutil_which("clang++"))
        argv: Argv = [
            f"--with-cc={cc}",
            f"--with-cxx={cxx}",
            "--build-type=debug",
            "--library-linkage=static",
            f"--cmake-generator={test_info.generator}",
            f"--CFLAGS={flags}",
            f"--CXXFLAGS={flags}",
        ]
        expected_spec: CMakeSpec = {
            "CMAKE_EXECUTABLE": f"{test_info.cmake_exe}",
            "CMAKE_GENERATOR": test_info.generator,
            "SOURCE_DIR": f"{AEDIFIX_PYTEST_DIR}",
            "BUILD_DIR": f"{test_info.cmake_dir}",
            "CMAKE_COMMANDS": [
                "--log-context",
                "--log-level=DEBUG",
                f"-DAEDIFIX_PYTEST_ARCH:STRING='{AEDIFIX_PYTEST_ARCH}'",
                f"-DAEDIFIX_PYTEST_DIR:PATH='{AEDIFIX_PYTEST_DIR}'",
                "-DBUILD_SHARED_LIBS:BOOL=OFF",
                "-DCMAKE_BUILD_TYPE:STRING=Debug",
                "-DCMAKE_COLOR_DIAGNOSTICS:BOOL=ON",
                "-DCMAKE_COLOR_MAKEFILE:BOOL=ON",
                f"-DCMAKE_CXX_COMPILER:FILEPATH={cxx}",
                f"-DCMAKE_CXX_FLAGS:STRING='{flags}'",
                f"-DCMAKE_C_COMPILER:FILEPATH={cc}",
                f"-DCMAKE_C_FLAGS:STRING='{flags}'",
                "-DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=ON",
            ],
        }

        ret = basic_configure(argv, DummyMainModule)
        assert ret == 0
        test_info.post_test(argv, expected_spec)


if __name__ == "__main__":
    sys.exit(pytest.main())
