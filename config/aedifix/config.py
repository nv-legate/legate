# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import re
import sys
from typing import TYPE_CHECKING, Final

from .base import Configurable
from .util.exception import UnsatisfiableConfigurationError
from .util.utility import cmake_configure_file

if TYPE_CHECKING:
    from pathlib import Path

    from .manager import ConfigurationManager


class ConfigFile(Configurable):
    r"""A helper class to manage a set of post-configuration config variables.
    These are written to disk after the configuration is complete so that other
    downstream tools may inspect or use them.

    Similar to a CMakeCache.txt.
    """

    __slots__ = (
        "_cmake_configure_file",
        "_config_file_template",
        "_default_subst",
    )

    def __init__(
        self, manager: ConfigurationManager, config_file_template: Path
    ) -> None:
        r"""Construct a Config.

        Parameters
        ----------
        manager : ConfigurationManager
            The configuration manager to manage this Config.
        config_file_template : Path
            The template file to read
        """
        super().__init__(manager=manager)
        self._config_file_template = config_file_template.resolve()
        self._default_subst = {"PYTHON_EXECUTABLE": sys.executable}

    @property
    def template_file(self) -> Path:
        r"""Return the path to the template file.

        Returns
        -------
        template_file : Path
            The path to the template file, e.g. /path/to/gmakevariables.in
        """
        return self._config_file_template

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

    def _read_entire_cmake_cache(self, cmake_cache: Path) -> dict[str, str]:
        r"""Read a CMakeCache.txt and convert all of the cache values to
        CMake command-line values in the form of -DNAME=VALUE.

        Parameters
        ----------
        cmake_cache : Path
            The path to the CMakeCache.txt.

        Returns
        -------
        list[str]
            A list of CMake command line arguments.
        """

        def keep_line(line: str) -> bool:
            line = line.strip()
            if not line:
                return False
            return not line.startswith(("//", "#"))

        cmake_variable_re: Final = re.compile(
            r"(?P<name>[A-Za-z_0-9\-]+):(?P<type>[A-Z]+)\s*=\s*(?P<value>.*)"
        )
        with cmake_cache.open() as fd:
            line_gen = (
                cmake_variable_re.match(line.lstrip())
                for line in filter(keep_line, fd)
            )
            return {m.group("name"): m.group("value") for m in line_gen if m}

    def _make_aedifix_substitutions(self, text: str) -> dict[str, str]:
        r"""Read the template file and find any aedifix-specific variable
        substitutions. Return a list of CMake command line arguments with the
        requested substitution value.

        Parameters
        ----------
        text : str
            The text of to the config file to parse.

        Returns
        -------
        list[str]
            The list of CMake commands.

        Raises
        ------
        UnsatisfiableConfigurationError
            If the substitution could not be made.
        """

        def make_subst(var: str) -> str | Path:
            try:
                return getattr(self.manager, var.casefold())
            except AttributeError:
                pass

            try:
                return self._default_subst[var]
            except KeyError:
                pass

            msg = f"Unknown project variable: {var!r}"
            raise UnsatisfiableConfigurationError(msg)

        ret = {}
        aedifix_vars = set(re.findall(r"@AEDIFIX_([^\s]+?)@", text))
        for var in aedifix_vars:
            value = str(make_subst(var))
            match value.casefold():
                case "on" | "yes" | "true":
                    value = "1"
                case "off" | "no" | "false":
                    value = "0"
                case _:
                    pass
            ret[f"AEDIFIX_{var}"] = value
        return ret

    def finalize(self) -> None:
        r"""Generate and dump project variables into the project variables
        file.

        Raises
        ------
        UnsatisfiableConfigurationError
            If the user config file contains an unknown AEDIFIX substitution.
        """
        project_file = self.project_variables_file
        template_file = self._config_file_template
        self.log(f"Using project file: {project_file}")
        self.log(f"Using template file: {template_file}")

        cache_vars = self.log_execute_func(
            self._read_entire_cmake_cache,
            self.project_cmake_dir / "CMakeCache.txt",
        )
        aedifix_vars = self.log_execute_func(
            self._make_aedifix_substitutions, template_file.read_text()
        )
        defs = cache_vars | aedifix_vars
        cmake_configure_file(self, template_file, project_file, defs)
        self.log(f"Wrote to project file: {project_file}")
