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

import inspect
import shutil
import stat
import sys
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Final

from .base import Configurable
from .util.utility import prune_command_line_args

if TYPE_CHECKING:
    from .manager import ConfigurationManager


class Reconfigure(Configurable):
    __slots__ = "_file", "_backup"

    TEMPLATE: Final = textwrap.dedent(
        r"""
        #!{PYTHON_EXECUTABLE}
        from __future__ import annotations

        import sys

        sys.path.insert(0, "{PROJECT_DIR}")

        from config.aedifix.main import basic_configure

        {MAIN_PACKAGE_IMPORT_LINE}


        def main() -> int:
            argv = [
        {ARGV_LIST}
            ] + sys.argv[1:]
            return basic_configure(
                tuple(argv), {MAIN_PACKAGE_TYPE}
            )

        if __name__ == "__main__":
            sys.exit(main())
        """
    ).strip()

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
    def _gen_import_line(main_package_type: type) -> str:
        main_package_module = inspect.getmodule(main_package_type)
        assert (
            main_package_module is not None
        ), "Could not determine module containing the main package!"
        if main_package_module.__package__:
            return (
                f"from {main_package_module.__name__} "
                f"import {main_package_type.__name__}"
            )
        return "\n".join(
            [
                "from config.configatus.util import load_module_from_path",
                "main_module = "
                f'load_module_from_path("{main_package_module.__file__}")',
            ]
        )

    def _sanitized_argv(self, ephemeral_args: set[str]) -> list[str]:
        pruned_cl_args = prune_command_line_args(
            self.manager.argv, ephemeral_args
        )
        # remove duplicates from the arguments
        seen = set()
        cl_args = []
        for f in pruned_cl_args:
            if f not in seen:
                seen.add(f)
                cl_args.append(f)
        # We want to include an explicit --PROJECT_ARCH=<whatever> in the
        # reconfigure script, in case the current project arch was taken via
        # environment variables.
        arch_flag = f"--{self.project_arch_name}"
        for arg in cl_args:
            if arch_flag in arg:
                break
        else:
            cl_args.insert(0, f'"{arch_flag}={self.project_arch}"')
        return cl_args

    def emit_file(self, text: str) -> None:
        r"""Emit the actual reconfigure file. Also deletes the backup, if
        any.

        Parameters
        ----------
        text : str
            The text to emit to the reconfigure file.
        """
        self.log(f"Generated reconfigure script:\n{text}")
        self.log(f"Writing reconfigure to {self.reconfigure_file}")
        self.reconfigure_file.write_text(text)
        old_st_mode = self.reconfigure_file.stat().st_mode
        self.log(
            "Making reconfigure script executable, "
            f"old permissions: {old_st_mode}"
        )
        self.reconfigure_file.chmod(old_st_mode | stat.S_IEXEC)
        self.log(
            "Making reconfigure script executable, "
            f"new permissions: {self.reconfigure_file.stat().st_mode}"
        )
        symlink = self.project_dir / self.reconfigure_file.name
        if symlink.exists():
            self.log(f"Reconfigure script symlink ({symlink}) already exists")
        else:
            self.log(f"Symlinking reconfigure script to {symlink}")
            symlink.symlink_to(
                self.reconfigure_file.relative_to(self.project_dir)
            )
        if self._backup is not None:
            self.log(
                f"Backup reconfigure script exists ({self._backup}), "
                "removing it!"
            )
            self._backup.unlink()
            self._backup = None

    def backup_reconfigure_script(self) -> None:
        r"""Create a backup of the reconfigure script for builds where
        --with-clean is specified, in case configure fails."""
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
        self, main_package_type: type, ephemeral_args: set[str]
    ) -> None:
        r"""Finalize the reconfigure script (i.e. instantiate it).

        Parameters
        ----------
        main_package_type : type
            The concrete type of the main package
        ephemeral_args : set[str]
            A set of arguments which appeared on the command line, but should
            not make it into the reconfigure script.

        Notes
        -----
        An example of an ephemeral arg is '--with-clean'. This argument should
        only be handled once; subsequent reconfigurations should not continue
        to delete and recreate the arch directory.
        """
        cl_args = self._sanitized_argv(ephemeral_args)
        template = self.TEMPLATE.format(
            PYTHON_EXECUTABLE=sys.executable,
            PROJECT_DIR=self.project_dir,
            MAIN_PACKAGE_IMPORT_LINE=self._gen_import_line(main_package_type),
            MAIN_PACKAGE_TYPE=main_package_type.__name__,
            ARGV_LIST="\n".join(f"        {arg}," for arg in cl_args),
        ).strip()
        self.emit_file(template)
