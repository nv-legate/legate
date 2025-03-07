# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys
import shutil
import inspect
from datetime import date
from typing import TYPE_CHECKING

from .base import Configurable
from .util.utility import (
    CMAKE_TEMPLATES_DIR,
    cmake_configure_file,
    deduplicate_command_line_args,
    prune_command_line_args,
)

if TYPE_CHECKING:
    from pathlib import Path

    from .manager import ConfigurationManager


class Reconfigure(Configurable):
    __slots__ = "_backup", "_file"

    def __init__(self, manager: ConfigurationManager) -> None:
        r"""Construct a Reconfigure.

        Parameters
        ----------
        manager : ConfigurationManager
            The configuration manager to manage this object.
        """
        super().__init__(manager=manager)
        fname = (
            f"reconfigure-{self.project_arch.replace(' ', '-').casefold()}.py"
        )
        self._file = self.project_arch_dir / fname
        self._backup: Path | None = None

    @property
    def reconfigure_file(self) -> Path:
        r"""Get the full path to the reconfigure file.

        Returns
        -------
        file : Path
            The path to the reconfigure file.
        """
        return self._file

    @staticmethod
    def get_import_line(main_package_type: type) -> tuple[str, str]:
        main_package_module = inspect.getmodule(main_package_type)
        assert main_package_module is not None, (
            "Could not determine module containing the main package!"
        )
        assert main_package_module.__package__
        return main_package_module.__name__, main_package_type.__name__

    def sanitized_argv(
        self,
        argv: tuple[str, ...],
        ephemeral_args: set[str],
        extra_argv: list[str] | None,
    ) -> list[str]:
        cl_args = prune_command_line_args(argv, ephemeral_args)
        cl_args = deduplicate_command_line_args(cl_args)
        # We want to include an explicit --PROJECT_ARCH=<whatever> in the
        # reconfigure script, in case the current project arch was taken via
        # environment variables.
        arch_flag = f"--{self.project_arch_name}"
        if not any(arg.startswith(arch_flag) for arg in cl_args):
            cl_args.insert(0, f"{arch_flag}={self.project_arch}")

        if extra_argv:
            if "--" not in cl_args:
                cl_args.append("--")
            cl_args.extend(extra_argv)
        return cl_args

    def backup_reconfigure_script(self) -> None:
        r"""Create a backup of the reconfigure script for builds where
        --with-clean is specified, in case configure fails.
        """
        symlink = self.project_dir / self.reconfigure_file.name
        self.log(f"Attempting to backup reconfigure script: {symlink}")
        if not symlink.exists():
            self.log(
                f"Reconfigure script symlink ({symlink}) does not exist, "
                "nothing to backup"
            )
            return

        self._backup = symlink.with_suffix(symlink.suffix + ".bk")
        self.log(f"Copying reconfigure script to backup: {self._backup}")
        shutil.copy2(symlink, self._backup, follow_symlinks=True)

    def finalize(  # type: ignore [override]
        self,
        main_package_type: type,
        ephemeral_args: set[str],
        extra_argv: list[str] | None = None,
    ) -> None:
        r"""Finalize the reconfigure script (i.e. instantiate it).

        Parameters
        ----------
        main_package_type : type
            The concrete type of the main package
        ephemeral_args : set[str]
            A set of arguments which appeared on the command line, but should
            not make it into the reconfigure script.
        extra_argv : list[str], optional
            Additional verbatim commands passed to CMake.

        Notes
        -----
        An example of an ephemeral arg is '--with-clean'. This argument should
        only be handled once; subsequent reconfigurations should not continue
        to delete and recreate the arch directory.
        """
        cl_args = self.sanitized_argv(
            self.manager.argv, ephemeral_args, extra_argv
        )
        cl_args_str = ",".join(f'"{arg}"' for arg in cl_args)

        mod_name, type_name = self.get_import_line(main_package_type)

        defs = {
            "PYTHON_EXECUTABLE": sys.executable,
            "YEAR": str(date.today().year),
            "PROJECT_DIR": self.project_dir,
            "MAIN_PACKAGE_MODULE": mod_name,
            "MAIN_PACKAGE_TYPE": type_name,
            "ARGV_LIST": cl_args_str,
        }

        self.log_execute_func(
            cmake_configure_file,
            self,
            CMAKE_TEMPLATES_DIR / "reconfigure_file.py.in",
            self.reconfigure_file,
            defs,
        )

        symlink = self.project_dir / self.reconfigure_file.name
        self.log(f"Symlinking reconfigure script to {symlink}")
        try:
            symlink.symlink_to(
                self.reconfigure_file.relative_to(self.project_dir)
            )
        except FileExistsError:
            self.log("Symlink destination already exists")

        if self._backup is not None:
            self.log(
                f"Backup reconfigure script exists ({self._backup}), "
                "removing it!"
            )
            self._backup.unlink()
            self._backup = None
