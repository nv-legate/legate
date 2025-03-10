# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import shlex
from typing import TYPE_CHECKING, Any, TypedDict, TypeVar

from ..util.exception import CMakeConfigureError, WrongOrderError
from .cmake_flags import CMakeList

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from ..manager import ConfigurationManager
    from ..package.main_package import DebugConfigureValue
    from .cmake_flags import CMakeFlagBase

    _T = TypeVar("_T")
    _CMakeFlagT = TypeVar("_CMakeFlagT", bound=CMakeFlagBase)


class CMakeCommandSpec(TypedDict):
    CMAKE_EXECUTABLE: str
    CMAKE_GENERATOR: str
    SOURCE_DIR: str
    BUILD_DIR: str
    CMAKE_COMMANDS: list[str]


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
            msg = (
                f"Variable {name} already registered as kind "
                f"{type(prev_reg)}, cannot overwrite it!"
            )
            raise ValueError(msg)  # noqa: TRY004
        manager.log(f"{name} already registered as kind {kind}")

    def _ensure_registered(self, name: str) -> None:
        if name not in self._args:
            msg = f"No variable with name {name!r} has been registered"
            raise WrongOrderError(msg)

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
        manager.log(
            f"Setting value {name} to {value} (current: {self._args[name]})"
        )
        self._args[name].value = value

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
        value = self._args[name].value
        manager.log(f"Value for {name}: {value}")
        return value

    def append_value(
        self, manager: ConfigurationManager, name: str, values: Sequence[_T]
    ) -> None:
        r"""Append a value to a CMake list.

        Parameters
        ----------
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
            msg = f"Cannot append to {type(cmake_var)}"
            raise TypeError(msg)
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
        manager: ConfigurationManager,
        cmd_spec: CMakeCommandSpec,
        cmd_file: Path,
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
            json.dump(cmd_spec, fd, sort_keys=True, indent=4)

    def _load_cmake_export_conf(self, manager: ConfigurationManager) -> None:
        conf_path = manager.project_export_config_path
        if not conf_path.is_file():
            m = f"CMake project failed to emit {conf_path}"
            raise CMakeConfigureError(m)

        config = json.loads(manager.project_export_config_path.read_text())
        for key, value in config.items():
            if value:
                self.set_value(manager, key, value)
            else:
                manager.log(
                    f"Ignoring cmake value {key} (falsey value: {value})"
                )

    def finalize(
        self,
        manager: ConfigurationManager,
        source_dir: Path,
        build_dir: Path,
        extra_argv: list[str] | None = None,
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
        extra_argv : list[str], optional
            Additional verbatim commands to pass to CMake.

        Raises
        ------
        CMakeConfigureError
            If the CMake configuration fails.
        """
        if extra_argv is None:
            extra_argv = []

        assert source_dir.exists(), "Source directory doesn't exist"
        manager.log(f"Using source dir: {source_dir}")
        manager.log(f"Using build dir: {build_dir}")
        manager.log(f"Using extra commands: {extra_argv}")
        if not build_dir.exists():
            build_dir.mkdir(parents=True)
        args = self._canonical_args()
        cmake_exe = args.pop("CMAKE_COMMAND").value
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

        def create_cmake_commands(*, quote: bool) -> list[str]:
            # These are the commands should go in the cmake_command.txt since
            # they are general for any invocation
            ret = ["--log-context", "--log-level=DEBUG"]
            debug_value: DebugConfigureValue = (
                manager.cl_args.debug_configure.value
            )
            ret.extend(debug_value.to_flags())
            ret.extend(
                (
                    "-DAEDIFIX:BOOL=ON",
                    f"-D{manager.project_arch_name}:STRING"
                    f"='{manager.project_arch}'",
                    f"-D{manager.project_dir_name}:PATH="
                    f"'{manager.project_dir}'",
                    f"-D{manager.project_name_upper}_CONFIGURE_OPTIONS:STRING="
                    f"{shlex.join(manager._orig_argv)}",  # noqa: SLF001
                )
            )
            export_vars = ";".join(arg.name for arg in self._args.values())

            ret.extend(
                (
                    f"-DAEDIFIX_EXPORT_VARIABLES:STRING='{export_vars}'",
                    "-DAEDIFIX_EXPORT_CONFIG_PATH:FILEPATH="
                    f"'{manager.project_export_config_path}'",
                )
            )

            ret.extend(
                value.to_command_line(quote=quote) for value in args.values()
            )

            # mypy is confused? We massage extra_argv into a list above
            assert isinstance(extra_argv, list)
            ret.extend(extra_argv)

            return ret

        cmake_extra_command = create_cmake_commands(quote=False)
        cmake_command = list(
            map(str, cmake_base_command + cmake_extra_command)
        )
        manager.log("Built CMake arguments:")
        manager.log("- " + "\n- ".join(cmake_command))

        cmd_spec: CMakeCommandSpec = {
            "CMAKE_EXECUTABLE": str(cmake_exe),
            "CMAKE_GENERATOR": generator.value,
            "SOURCE_DIR": str(source_dir),
            "BUILD_DIR": str(build_dir),
            "CMAKE_COMMANDS": create_cmake_commands(quote=True),
        }

        self._dump_cmake_command_spec(
            manager, cmd_spec, build_dir / "aedifix_cmake_command_spec.json"
        )

        manager.log_boxed(
            "This may take a few minutes",
            title=f"Configuring {manager.project_name}",
        )
        try:
            manager.log_execute_command(cmake_command, live=True)
        except Exception as e:
            msg = f"CMake failed to configure {manager.project_name}"
            raise CMakeConfigureError(msg) from e

        manager.log_divider(tee=True)
        self._load_cmake_export_conf(manager)
