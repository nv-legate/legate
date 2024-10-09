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
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Final

from .base import Configurable
from .util.exception import UnsatisfiableConfigurationError

if TYPE_CHECKING:
    from .manager import ConfigurationManager


class ConfigFile(Configurable):
    r"""A helper class to manage a set of post-configuration config variables.
    These are written to disk after the configuration is complete so that other
    downstream tools may inspect or use them.

    Similar to a CMakeCache.txt.
    """

    __slots__ = (
        "_config_file_template",
        "_cmake_configure_file",
        "_default_subst",
    )

    def __init__(
        self,
        manager: ConfigurationManager,
        config_file_template: Path,
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

        self._cmake_configure_file = (
            Path(__file__).resolve().parent / "configure_file.cmake"
        )
        assert (
            self._cmake_configure_file.exists()
        ), f"Cmake configure file {self._cmake_configure_file} does not exist"
        assert (
            self._cmake_configure_file.is_file()
        ), f"Cmake configure file {self._cmake_configure_file} is not a file"

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

    @property
    def cmake_configure_file(self) -> Path:
        r"""Return the cmake configure file to use to generate the
        project config file.

        Returns
        -------
        configure_file : Path
            The path to the cmake configure file.
        """
        return self._cmake_configure_file

    def _cmake_cache_to_cmd_line_args(self, cmake_cache: Path) -> list[str]:
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
            if line.startswith(("//", "#")):
                return False
            return True

        cmake_variable_re: Final = re.compile(
            r"(?P<name>[A-Za-z_0-9\-]+):(?P<type>[A-Z]+)\s*=\s*(?P<value>.*)"
        )
        with cmake_cache.open() as fd:
            line_gen = (
                cmake_variable_re.match(line.lstrip())
                for line in filter(keep_line, fd)
            )
            lines = [
                f"-D{m.group('name')}={m.group('value')}"
                for m in line_gen
                if m
            ]

        return lines

    def _make_aedifix_substitutions(self, text: str) -> list[str]:
        r"""Read the template file and find any aedifix-specific variable
        subsitutions. Return a list of CMake command line arguments with the
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

            raise UnsatisfiableConfigurationError(
                f"Unknown project variable: {var!r}"
            )

        ret = []
        aedifix_vars = set(re.findall(r"@AEDIFIX_([^\s]+?)@", text))
        for var in aedifix_vars:
            value = make_subst(var)
            ret.append(f"-DAEDIFIX_{var}={value}")
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

        cmake_exe = self.manager.read_cmake_variable("CMAKE_COMMAND")
        cache_vars = self.log_execute_func(
            self._cmake_cache_to_cmd_line_args,
            self.project_cmake_dir / "CMakeCache.txt",
        )
        aedifix_vars = self.log_execute_func(
            self._make_aedifix_substitutions, template_file.read_text()
        )
        rest_vars = [
            f"-DAEDIFIX_CONFIGURE_FILE_SRC={template_file}",
            f"-DAEDIFIX_CONFIGURE_FILE_DEST={project_file}",
            "-P",
            self.cmake_configure_file,
        ]
        self.log_execute_command(
            [cmake_exe] + cache_vars + aedifix_vars + rest_vars
        )
        self.log(f"Wrote to project file: {project_file}")
