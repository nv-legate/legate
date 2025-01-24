# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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
import json
import shlex
from typing import TypedDict

from ._io import vprint, warning_print
from ._legate_config import get_legate_config


class CMakeSpec(TypedDict):
    CMAKE_GENERATOR: str
    CMAKE_COMMANDS: list[str]


class CMakeConfig:
    def __init__(self) -> None:
        # We wish to configure the build using exactly the same arguments as
        # the C++ lib so that ./configure options are respected.
        cmd_spec_path = (
            get_legate_config().LEGATE_CMAKE_DIR
            / "aedifix_cmake_command_spec.json"
        )
        vprint(f"Using cmake_command file: {cmd_spec_path}")
        if cmd_spec_path.exists():
            with cmd_spec_path.open() as fd:
                cmake_spec: CMakeSpec = json.load(fd)
        else:
            # User has not run configure, and so the spec file does not
            # exist. Default to
            cmake_spec = {
                "CMAKE_GENERATOR": os.environ.get(
                    "CMAKE_GENERATOR", "Unix Makefiles"
                ),
                "CMAKE_COMMANDS": [],
            }
            warning_print(
                "Running pip install without first configuring legate. "
                f"Using an default-generated command spec:\n{cmake_spec}"
            )

        cmake_args = self._read_cmake_args(cmake_spec)
        cmake_args, cmake_defines = self._split_out_cmake_defines(cmake_args)

        try:
            build_type = cmake_defines["CMAKE_BUILD_TYPE"]
        except KeyError:
            build_type = "Release"
            vprint(f"Using default build type: {build_type}")
        else:
            vprint(f"Found build type: {build_type}")

        self._build_type = build_type
        self._cmake_args = cmake_args
        self._cmake_defines = cmake_defines
        self._generator = cmake_spec["CMAKE_GENERATOR"]

    @staticmethod
    def _read_cmake_args(cmake_spec: CMakeSpec) -> list[str]:
        def read_env_args(name: str) -> list[str]:
            # Always use shlex to split, since some CMake variables are
            # semi-colon separated internally. For example:
            # '-DSOME_CMAKE_PATHS=/foo/bar;/baz/bop' should be parsed as a
            # single value, not 2.
            args = (x.strip() for x in shlex.split(os.environ.get(name, "")))
            return [x for x in args if x]

        cmake_args = [
            arg
            for arg in cmake_spec["CMAKE_COMMANDS"]
            if "CMAKE_INSTALL_PREFIX" not in arg
        ]
        vprint(f"Initialized cmake args from command spec: {cmake_args}")

        for name in ("CMAKE_ARGS", "SKBUILD_CMAKE_ARGS"):
            env_args = read_env_args(name)
            vprint(f"Adding {name} to cmake_args: {env_args}")
            cmake_args.extend(env_args)

        return cmake_args

    @staticmethod
    def _split_out_cmake_defines(
        cmake_args: list[str],
    ) -> tuple[list[str], dict[str, str]]:
        cmake_defines = {}
        new_cmake_args = []

        for arg in cmake_args:
            if arg.startswith("-D"):
                if "=" not in arg:
                    msg = (
                        f"CMake define {arg!r} not in the form '-DNAME=value'"
                    )
                    raise ValueError(msg)
                name, _, value = arg.partition("=")
                if name.count(":") > 1:
                    msg = (
                        "Too many colons (:) in {arg!r}. This may be correct "
                        "in principle, but the build system expects only 1."
                    )
                    raise ValueError(msg)
                name = name.removeprefix("-D")
                name = name.partition(":")[0]
                cmake_defines[name] = value
            else:
                new_cmake_args.append(arg)
        return new_cmake_args, cmake_defines

    @property
    def cmake_defines(self) -> dict[str, str]:
        return self._cmake_defines

    @property
    def cmake_args(self) -> list[str]:
        return self._cmake_args

    @property
    def build_type(self) -> str:
        return self._build_type

    @property
    def generator(self) -> str:
        return self._generator
