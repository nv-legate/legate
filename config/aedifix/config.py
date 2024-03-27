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
import sys
import textwrap
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final

from .base import Configurable
from .util.exception import LengthError

if TYPE_CHECKING:
    from .manager import ConfigurationManager


CMAKE_VARIABLE_RE: Final = re.compile(
    r"(?P<name>[A-Za-z_0-9\-]+):(?P<type>[A-Z]+)\s*=\s*(?P<value>.*)"
)


class ConfigFile(Configurable):
    r"""A helper class to manage a set of post-configuration config variables.
    These are written to disk after the configuration is complete so that other
    downstream tools may inspect or use them.

    Similar to a CMakeCache.txt.
    """

    __slots__ = (
        "_project_rules",
        "_project_search_variables",
        "_project_variables",
    )

    def __init__(self, manager: ConfigurationManager) -> None:
        r"""Construct a Config.

        Parameters
        ----------
        manager : ConfigurationManager
            The configuration manager to manage this Config.
        """
        super().__init__(manager=manager)
        self._project_rules: dict[
            str, tuple[bool, Sequence[str], tuple[str, ...]]
        ] = {}
        self._project_search_variables: dict[str, str] = {}
        self._project_variables: dict[str, tuple[bool, str]] = {}

    @property
    def project_variables_file(self) -> Path:
        r"""Return the project variables file.

        Returns
        -------
        variables_file : Path
            The full path to the project variables file.

        Notes
        -----
        The file is not guaranteed to exist, or be up to date. Usually it is
        created/refreshed during finalization of this object.
        """
        return self.project_arch_dir / "gmakevariables"

    def _add_default_project_rules(self, PROJ_NAME: str) -> None:
        r"""Add default project rules.

        Parameters
        ----------
        PROJ_NAME : str
            The uppercase project name, i.e. 'LEGATE_CORE'.
        """
        assert PROJ_NAME.isupper()
        self.add_rule(
            "default_help",
            "printf \"Usage: make [MAKE_OPTIONS] [target] (see 'make --help' "
            'for MAKE_OPTIONS)\\n"',
            'printf ""',
            textwrap.dedent(
                r"""
    $(AWK) ' \
    { \
      if ($$0 ~ /^.PHONY: [a-zA-Z\-\0-9]+$$/) {	\
        helpCommand = substr($$0, index($$0, ":") + 2);	\
        if (helpMessage) { \
          printf "\033[36m%-20s\033[0m %s\n", helpCommand, helpMessage; \
          helpMessage = ""; \
        } \
      } else if ($$0 ~ /^[a-zA-Z\-\0-9.]+:/) { \
        helpCommand = substr($$0, 0, index($$0, ":")); \
        if (helpMessage) { \
          printf "\033[36m%-20s\033[0m %s\n", helpCommand, helpMessage; \
          helpMessage = ""; \
        } \
      } else if ($$0 ~ /^##/) { \
        if (helpMessage) { \
          helpMessage = helpMessage"\n                     "substr($$0, 3); \
        } else { \
          helpMessage = substr($$0, 3); \
        } \
      } else { \
        if (helpMessage) { \
          print "\n                     "helpMessage"\n"; \
        } \
        helpMessage = ""; \
      } \
    }' \
    $(MAKEFILE_LIST)
            """.strip()
            ),
        )
        self.add_rule(
            "default_all",
            f"$({PROJ_NAME}_BUILD_COMMAND) $({PROJ_NAME}_CMAKE_ARGS)",
        )
        self.add_rule(
            "default_clean",
            f"$({PROJ_NAME}_BUILD_COMMAND) "
            "--target clean "
            f"$({PROJ_NAME}_CMAKE_ARGS)",
        )
        self.add_rule(
            "default_install",
            f"$({PROJ_NAME}_INSTALL_COMMAND) $({PROJ_NAME}_CMAKE_ARGS)",
        )
        self.add_rule(
            "default_package",
            f"$({PROJ_NAME}_BUILD_COMMAND) "
            "--target package "
            f"$({PROJ_NAME}_CMAKE_ARGS)",
        )

    def _add_default_project_variables(
        self, PROJ_ARCH_NAME: str, PROJ_NAME: str
    ) -> None:
        r"""Add default project project variables.

        Parameters
        ----------
        PROJ_ARCH_NAME : str
            The uppercase project name, i.e. 'LEGATE_CORE_ARCH'.
        PROJ_NAME : str
            The uppercase project name, sans the 'ARCH' i.e. 'LEGATE_CORE'.
        """
        assert PROJ_ARCH_NAME.isupper()
        assert PROJ_NAME.isupper()
        # unconditional variables
        self.add_variable("PYTHON", sys.executable, override_ok=True)
        self.add_variable("SHELL", "/bin/sh", override_ok=True)
        self.add_variable("AWK", "awk", override_ok=True)
        self.add_variable("CP", "cp", override_ok=True)
        self.add_variable("MV", "mv", override_ok=True)
        self.add_variable("CMAKE", "cmake", override_ok=True)
        PROJ_DIR_NAME = self.project_dir_name.upper()
        self.add_variable(
            f"{PROJ_NAME}_BUILD_COMMAND",
            f"$(CMAKE) "
            f"--build $({PROJ_DIR_NAME})/$({PROJ_ARCH_NAME})/cmake_build ",
        )
        self.add_variable(
            f"{PROJ_NAME}_INSTALL_COMMAND",
            f"$(CMAKE) "
            f"--install $({PROJ_DIR_NAME})/$({PROJ_ARCH_NAME})/cmake_build ",
        )

        # search variables
        self.add_search_variable(
            "Python3_EXECUTABLE", project_var_name="PYTHON"
        )

    def _finalize_project_search_variables(self, cache_file: Path) -> None:
        r"""Search the cache file and populate the project variables from
        search.

        Parameters
        ----------
        cache_file : Path
            The path to the CMakeCache.txt file to search.

        Notes
        -----
        This fills `self._project_variables` with variables found from
        `self._project_search_variables`.
        """

        def found_variable(
            name: str, value: str, cmake_name: str, project_var_name: str
        ) -> bool:
            found = name == cmake_name
            if found:
                self.log(f"Found variable: {name}")
                self.add_variable(project_var_name, value)
            return found

        self.log(f"Reading cache file: {cache_file}")
        assert cache_file.exists(), f"{cache_file} does not exist!"
        with cache_file.open() as fd:
            for line in filter(None, map(str.strip, fd)):
                if line.startswith("//") or line.startswith("#"):
                    # ignore cmake-comment lines entirely
                    continue
                matched = CMAKE_VARIABLE_RE.match(line)
                assert (
                    matched is not None
                ), f"Did not find cmake variable match for line: {line}"
                name = matched.group("name")
                value = matched.group("value")
                for (
                    cmake_name,
                    project_var_name,
                ) in self._project_search_variables.items():
                    if found_variable(
                        name, value, cmake_name, project_var_name
                    ):
                        continue

    def _finalize_project_variables(self) -> list[str]:
        r"""Finalize the project variables.

        Returns
        -------
        variables : list[str]
            The list of variable declaration lines, to be appended to the
            global list.
        """
        self.log_execute_func(
            self._finalize_project_search_variables,
            self.project_cmake_dir / "CMakeCache.txt",
        )
        return [
            f"{key.upper()} ?= {value}"
            for key, (_, value) in self._project_variables.items()
        ]

    def _finalize_project_rules(self) -> list[str]:
        r"""Finalize the project rules.

        Returns
        -------
        rules : list[str]
            The list of rule lines, to be appended to the global list.
        """
        lines = []
        for rule_name, (
            phony,
            deps,
            rule_lines,
        ) in self._project_rules.items():
            lines.append("")
            if phony:
                lines.append(f".PHONY: {rule_name}")
            depstr = " " + " ".join(deps)
            lines.extend(
                [f"{rule_name}:{depstr}", "\t@" + "\n\t@".join(rule_lines)]
            )
        return lines

    def add_rule(
        self,
        rule_name: str,
        *rule_lines: str,
        phony: bool = True,
        deps: Sequence[str] | None = None,
        exist_ok: bool = False,
    ) -> None:
        r"""Add a project rule.

        Parameters
        ----------
        rule_name : str
            The name of the project rule to add.
        *rule_lines : str
            The lines comprising the body of the rule.
        phony : bool, True
            Whether the rule is .PHONY.
        deps : Sequence[str], optional
            Dependencies of the rule (if any).
        exist_ok : bool, False
            True if the rule should be overriden, False if this is an error.

        Raises
        ------
        LengthError
            If `rule_name` is a zero length.
        ValueError
            If the rule exists and `exist_ok` is False.
        """
        rule_name = rule_name.strip()
        if not rule_name:
            raise LengthError("Rule name must not be empty")

        if deps is None:
            deps = tuple()

        if not rule_lines and not len(deps):
            raise ValueError("Cannot have an empty rule with empty deps!")

        self.log(f"Adding config file rule: {rule_name}")
        self.log(f"project rule '{rule_name}' deps: {deps}")
        self.log(f"project rule '{rule_name}' lines: {rule_lines}")
        if rule_name in self._project_rules:
            if not exist_ok:
                raise ValueError(f"Project rule '{rule_name}' already exists")
            self.log(
                f"Project rule '{rule_name}' already exists, will be "
                f"overridden! {self._project_rules[rule_name]}"
            )
        else:
            self.log(f"Project rule '{rule_name}' does not exist, adding it")
        self._project_rules[rule_name] = (phony, deps, rule_lines)

    def add_variable(
        self, var_name: str, value: str, override_ok: bool = False
    ) -> None:
        r"""Add a project variable.

        Parameters
        ----------
        var_name : str
            The name of the project variable.
        value : str
            The value of the project variable.
        override_ok : bool, False
            Whether the value of the variable may be overriden at a later date.

        Raises
        ------
        LengthError
            If `var_name` is empty.
        ValueError
            If the variable exist and was previously registered with
            `override_ok` = False.
        """
        var_name = var_name.strip()
        if not var_name:
            raise LengthError("Variable name must not be empty")

        try:
            old_ovr_ok, old_val = self._project_variables[var_name]
        except KeyError:
            self.log(
                f"Adding project variable: {var_name} = "
                f"({override_ok = }, {value = })"
            )
        else:
            if not old_ovr_ok:
                raise ValueError(
                    f"Project variable {var_name} already registered:"
                    f" ({old_ovr_ok = }, {old_val = })"
                )
            self.log(
                f"project variable '{var_name}' already exists, will be "
                f"overriden! New value = ({override_ok = }, {value = })"
            )
        self._project_variables[var_name] = (override_ok, value)

    def add_search_variable(
        self,
        cmake_name: str,
        project_var_name: str | None = None,
        exist_ok: bool = False,
    ) -> None:
        r"""Add a variable whose value should be searched for in the
        cmake cache after cmake has run.

        Parameters
        ----------
        cmake_name : str
            The name of the variable in the CMakeCache.txt.
        project_var_name : str, optional
            The name the variable should have in the projectvariables file.
            Defaults to `cmake_name` if not given.
        exist_ok : bool, False
            True if the variable can already exist, False if this is an error.

        Raises
        ------
        LengthError
            If either `cmake_name` is empty, or (if given) `project_var_name`
            is empty.
        ValueError
            If the the variable exists and `exist_ok` is False.
        """
        cmake_name = cmake_name.strip()
        if not cmake_name:
            raise LengthError("CMake name must not be empty")

        if project_var_name is None:
            project_var_name = cmake_name
        else:
            project_var_name = project_var_name.strip()
            if not project_var_name:
                raise LengthError("Project variable name must not be empty")

        try:
            old_project_var_name = self._project_search_variables[cmake_name]
        except KeyError:
            self.log(
                f"Adding project search variable: {cmake_name} -> "
                f"{project_var_name}"
            )
        else:
            if not exist_ok:
                raise ValueError(
                    f"Project search variable {cmake_name} already registered"
                )
            self.log(
                f"project search variable {cmake_name} -> "
                f"{old_project_var_name} already exists, will be overriden!"
            )
        self._project_search_variables[cmake_name] = project_var_name

    def setup(self) -> None:
        r"""Setup the config.

        Notes
        -----
        This populates the config with the default rules and variables.
        """
        PROJ_ARCH_NAME = self.project_arch_name.upper()
        PROJ_NAME = PROJ_ARCH_NAME.replace("_ARCH", "")

        self.log_execute_func(self._add_default_project_rules, PROJ_NAME)
        self.log_execute_func(
            self._add_default_project_variables, PROJ_ARCH_NAME, PROJ_NAME
        )

    def finalize(self) -> None:
        r"""Generate and dump project variables into the project variables
        file.
        """
        header_lines = [
            r"# -*- mode: makefile-gmake -*-",
            f"# WARNING: this file was generated by {__file__}.",
            r"# Any modifications may be lost when configure is next invoked!",
            r"",
            r"MAKEFLAGS += --no-builtin-rules",
        ]
        project_variables = self.log_execute_func(
            self._finalize_project_variables
        )
        project_rules = self.log_execute_func(self._finalize_project_rules)

        project_file = self.project_variables_file
        self.log(f"Using project file: {project_file}")
        project_file.write_text(
            "\n".join(header_lines + project_variables + project_rules)
        )
        self.log(f"Wrote to project file: {project_file}")
