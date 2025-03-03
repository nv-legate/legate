# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, ParamSpec, TypeVar

from .logger import Logger

if TYPE_CHECKING:
    from argparse import Namespace
    from collections.abc import Callable, Sequence
    from pathlib import Path
    from subprocess import CompletedProcess

    from .logger import AlignMethod
    from .manager import ConfigurationManager

_P = ParamSpec("_P")
_T = TypeVar("_T")


class Configurable:
    __slots__ = ("_manager",)

    def __init__(self, manager: ConfigurationManager) -> None:
        r"""Construct a Configurable.

        Parameters
        ----------
        manager : ConfigurationManager
            The global configuration manager which manages this configurable.
        """
        self._manager = manager

    @property
    def manager(self) -> ConfigurationManager:
        r"""Get the configuration manager.

        Returns
        -------
        manager : ConfigurationManager
            The configuration manager that manages this configurable.
        """
        return self._manager

    @property
    def cl_args(self) -> Namespace:
        r"""See `ConfigurationManager.cl_args`."""
        return self.manager.cl_args

    @property
    def project_name(self) -> str:
        r"""See `ConfigurationManager.project_name`."""
        return self.manager.project_name

    @property
    def project_name_upper(self) -> str:
        r"""See `ConfigurationManager.project_name`."""
        return self.manager.project_name_upper

    @property
    def project_arch(self) -> str:
        r"""See `ConfigurationManager.project_arch`."""
        return self.manager.project_arch

    @property
    def project_arch_name(self) -> str:
        r"""See `ConfigurationManager.project_arch_name`."""
        return self.manager.project_arch_name

    @property
    def project_dir(self) -> Path:
        r"""See `ConfigurationManager.project_dir`."""
        return self.manager.project_dir

    @property
    def project_src_dir(self) -> Path:
        r"""See `ConfigurationManager.project_src_dir`."""
        return self.manager.project_src_dir

    @property
    def project_dir_name(self) -> str:
        r"""See `ConfigurationManager.project_dir_name`."""
        return self.manager.project_dir_name

    @property
    def project_arch_dir(self) -> Path:
        r"""See `ConfigurationManager.project_arch_dir`."""
        return self.manager.project_arch_dir

    @property
    def project_cmake_dir(self) -> Path:
        r"""See `ConfigurationManager.project_cmake_dir`."""
        return self.manager.project_cmake_dir

    @Logger.log_passthrough
    def log(
        self,
        msg: str | list[str] | tuple[str, ...],
        *,
        tee: bool = False,
        caller_context: bool = True,
        keep: bool = False,
    ) -> None:
        r"""See `ConfigurationManager.log`."""
        return self.manager.log(
            msg, tee=tee, caller_context=caller_context, keep=keep
        )

    @Logger.log_passthrough
    def log_divider(self, *, tee: bool = False, keep: bool = False) -> None:
        r"""See `ConfigurationManager.log_divider`."""
        return self.manager.log_divider(tee=tee, keep=keep)

    @Logger.log_passthrough
    def log_boxed(
        self,
        message: str,
        *,
        title: str = "",
        title_style: str = "",
        align: AlignMethod = "center",
    ) -> None:
        r"""See `ConfigurationManager.log_boxed`."""
        return self.manager.log_boxed(
            message, title=title, title_style=title_style, align=align
        )

    @Logger.log_passthrough
    def log_warning(self, message: str, *, title: str = "WARNING") -> None:
        r"""See `ConfigurationManager.log_warning`."""
        return self.manager.log_warning(message, title=title)

    @Logger.log_passthrough
    def log_execute_func(
        self, fn: Callable[_P, _T], *args: _P.args, **kwargs: _P.kwargs
    ) -> _T:
        r"""See `ConfigurationManager.log_execute_func`."""
        return self.manager.log_execute_func(fn, *args, **kwargs)

    @Logger.log_passthrough
    def log_execute_command(
        self, command: Sequence[_T], *, live: bool = False
    ) -> CompletedProcess[str]:
        r"""See `ConfigurationManager.log_execute_command`."""
        return self.manager.log_execute_command(command, live=live)

    def setup(self) -> None:
        r"""Setup a `Configurable` for later configuration. By default,
        does nothing.
        """

    def configure(self) -> None:
        r"""Configure a `Configurable`, setting any options. By default, does
        nothing.
        """

    def finalize(self) -> None:
        r"""Finalize a `Configurable`. By default, does nothing."""
