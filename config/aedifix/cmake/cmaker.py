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
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from ..util.exception import CMakeConfigureError, WrongOrderError
from .cmake_flags import CMakeList

if TYPE_CHECKING:
    from ..manager import ConfigurationManager
    from .cmake_flags import CMakeFlagBase

    _T = TypeVar("_T")
    _CMakeFlagT = TypeVar("_CMakeFlagT", bound=CMakeFlagBase)


class CMaker:
    __slots__ = ("_args",)

    def __init__(self) -> None:
        r"""Construct a CMaker."""
        self._args: dict[str, CMakeFlagBase] = {}

    def register_variable(
        self, manager: ConfigurationManager, var: _CMakeFlagT
    ) -> None:
        r"""Register a CMake variable.

        Parameters
        ----------
        manager : ConfigurationManager
            The manager responsible for the current configuration.
        var : CMakeFlagT
            The variable to register.

        Raises
        ------
        ValueError
            If `var` was previously registered with a different type.
        """
        kind = type(var)
        name = var.name
        manager.log(f"Trying to register {name} as kind {kind}")
        if name not in self._args:
            self._args[name] = var
            manager.log(f"Successfully registered {name} as kind {kind}")
            return

        prev_reg = self._args[name]
        if not isinstance(prev_reg, kind):
            raise ValueError(
                f"Variable {name} already registered as kind "
                f"{type(prev_reg)}, cannot overwrite it!"
            )
        manager.log(f"{name} already registered as kind {kind}")

    def _ensure_registered(self, name: str) -> None:
        if name not in self._args:
            raise WrongOrderError(
                f"No variable with name {name!r} has been registered"
            )

    def set_value(
        self, manager: ConfigurationManager, name: str, value: _T
    ) -> None:
        r"""Set a CMake variable's value.

        Parameters
        ----------
        manager : ConfigurationManager
            The manager responsible for the current configuration.
        name : str
            The name of the CMake variable.
        value : T
            The value to set the variable to.

        Raises
        ------
        WrongOrderError
            If no variable with name `name` has been registered.
        """
        self._ensure_registered(name)
        manager.log(f"Setting value {name} to {value}")
        manager.log(f"Current value for {name}: {self._args[name]}")
        self._args[name].value = value
        manager.log(f"New value for {name}: {value}")

    def get_value(self, manager: ConfigurationManager, name: str) -> Any:
        r"""Get a CMake variable's value.

        Parameters
        ----------
        manager : ConfigurationManager
            The manager responsible for the current configuration.
        name : str
            The name of the CMake variable.

        Returns
        -------
        value : Any
            The value of the CMake variable.

        Raises
        ------
        WrongOrderError
            If no variable with name `name` has been registered.
        """
        self._ensure_registered(name)
        manager.log(f"Getting value of {name}")
        value = self._args[name].value
        manager.log(f"Value for {name}: {value}")
        return value

    def append_value(
        self, manager: ConfigurationManager, name: str, values: Sequence[_T]
    ) -> None:
        r"""Append a value to a CMake list.

        Paramters
        ---------
        manager : ConfigurationManager
            The manager responsible for the current configuration.
        name : str
            The name of the CMake variable.
        values : Sequence[str]
            The values to append to the list.

        Raises
        ------
        WrongOrderError
            If no variable with name `name` has been registered.
        TypeError
            If the CMake variable is not a list-type.
        """
        self._ensure_registered(name)
        manager.log(f"Appending values {values} to {name}")
        if not values:
            manager.log("No values to append, bailing")
            return

        cmake_var = self._args[name]
        if not isinstance(cmake_var, CMakeList):
            raise TypeError(f"Cannot append to {type(cmake_var)}")
        cur_values = cmake_var.value
        # Need "was_none" since the getter/setter for cmake_var may perform a
        # copy on assignment, so cmake_var.value = cur_values = [] wouldn't
        # work, since cmake_var.value would not contain the same list object as
        # cur_values.
        if was_none := (cur_values is None):
            cur_values = []
        else:
            assert isinstance(cur_values, list)

        manager.log(f"Current values for {name}: {cur_values}")
        cur_values.extend(values)
        manager.log(f"New values for {name}: {cur_values}")
        if was_none:
            cmake_var.value = cur_values

    def _canonical_args(self) -> dict[str, CMakeFlagBase]:
        ret = {}
        for key, value in self._args.items():
            canonical = value.canonicalize()
            if canonical is None:
                continue
            ret[key] = canonical
        return ret

    @staticmethod
    def _dump_cmake_command_spec(
        manager: ConfigurationManager, cmd_spec: dict[str, Any], cmd_file: Path
    ) -> None:
        if cmd_file.exists():
            manager.log(f"Command file {cmd_file} already exists, loading it")
            with cmd_file.open() as fd:
                old_cmds = json.load(fd)["CMAKE_COMMANDS"]

            new_cmds = cmd_spec["CMAKE_COMMANDS"]
            # De-deuplicate keys. Note this won't catch everything,
            # i.e. --foo=1 --foo=2 won't get deduplicated, but at least the
            # ordering is kept.
            new_cmds[:] = list(dict.fromkeys(old_cmds + new_cmds))
            manager.log(f"Merged and de-duplicated cmake commands: {new_cmds}")

        manager.log(f"Saving configure command to {cmd_file}")
        with cmd_file.open("w") as fd:
            json.dump(cmd_spec, fd)

    def finalize(
        self, manager: ConfigurationManager, source_dir: Path, build_dir: Path
    ) -> None:
        r"""Execute the CMake configuration.

        Parameters
        ----------
        manager : ConfigurationManager
            The configuration to execute.
        source_dir : Path
            The full path to the source directory of the project. Usually
            synonymous with the project directory.
        build_dir : Path
            The full path to the build directory in which to invoke cmake.
        """
        assert source_dir.exists(), "Source directory doesn't exist"
        manager.log(f"Using source dir: {source_dir}")
        manager.log(f"Using build dir: {build_dir}")
        if not build_dir.exists():
            build_dir.mkdir(parents=True)
        args = self._canonical_args()
        cmake_exe = args.pop("CMAKE_EXECUTABLE").value
        generator = args.pop("CMAKE_GENERATOR")
        # These commands should not go in the cmake_command.txt since they
        # pertain only to this precise invocation.
        cmake_base_command = [
            cmake_exe,
            "-S",
            source_dir,
            "-B",
            build_dir,
            generator.prefix,
            generator.value,
        ]

        def create_cmake_extra_commands(quote: bool) -> list[str]:
            # These are the commands should go in the cmake_command.txt since
            # they are general for any invocation
            return [
                "--log-context",
                "--log-level=DEBUG",
                f"-D{manager.project_arch_name}:STRING"
                f"='{manager.project_arch}'",
                f"-D{manager.project_dir_name}:PATH='{manager.project_dir}'",
            ] + [value.to_command_line(quote=quote) for value in args.values()]

        cmake_extra_command = create_cmake_extra_commands(quote=False)
        cmake_command = list(
            map(str, cmake_base_command + cmake_extra_command)
        )
        manager.log("Built CMake arguments:")
        manager.log("- " + "\n- ".join(cmake_command))

        cmd_spec = {
            "CMAKE_EXECUTABLE": str(cmake_exe),
            "CMAKE_GENERATOR": generator.value,
            "SOURCE_DIR": str(source_dir),
            "BUILD_DIR": str(build_dir),
            "CMAKE_COMMANDS": create_cmake_extra_commands(quote=True),
        }

        self._dump_cmake_command_spec(
            manager, cmd_spec, build_dir / "aedifix_cmake_command_spec.json"
        )

        manager.log_boxed(
            "This may take a few minutes",
            title=f"Configuring {manager.project_name}",
        )
        try:
            manager.log_execute_command(cmake_command)
        except Exception as e:
            raise CMakeConfigureError(
                f"CMake failed to configure {manager.project_name}"
            ) from e

        manager.log_divider()
