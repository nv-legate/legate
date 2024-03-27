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

from argparse import Namespace
from collections.abc import Callable, Sequence
from pathlib import Path
from subprocess import CompletedProcess
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from .logger import Logger

if TYPE_CHECKING:
    from .manager import ConfigurationManager

_P = ParamSpec("_P")
_T = TypeVar("_T")


class Configurable:
    __slots__ = "_manager"

    def __init__(self, manager: ConfigurationManager) -> None:
        r"""Construct a Configurable.

        Parameters
        ----------
        manager : ConfigurationManager
            The global configuration manager which manages this configurable.
        """
        self._manager = manager
        super().__init__()

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
    def project_name(self) -> str:
        r"""See `ConfigurationManager.project_name`."""
        return self.manager.project_name

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

    @property
    def cl_args(self) -> Namespace:
        r"""See `ConfigurationManager.cl_args`."""
        return self.manager.cl_args

    @Logger.log_passthrough
    def log(
        self,
        mess: str,
        *,
        tee: bool = False,
        end: str = "\n",
        scroll: bool = False,
        caller_context: bool = True,
    ) -> None:
        r"""See `ConfigurationManager.log`."""
        return self.manager.log(
            mess,
            tee=tee,
            end=end,
            scroll=scroll,
            caller_context=caller_context,
        )

    @Logger.log_passthrough
    def log_divider(self, tee: bool = False) -> None:
        r"""See `ConfigurationManager.log_divider`."""
        return self.manager.log_divider(tee=tee)

    @Logger.log_passthrough
    def log_boxed(
        self,
        message: str,
        *,
        title: str = "",
        divider_char: str | None = None,
        tee: bool = True,
        caller_context: bool = False,
        **kwargs: Any,
    ) -> None:
        r"""See `ConfigurationManager.log_boxed`."""
        return self.manager.log_boxed(
            message,
            title=title,
            divider_char=divider_char,
            tee=tee,
            caller_context=caller_context,
            **kwargs,
        )

    @Logger.log_passthrough
    def log_warning(
        self, message: str, *, title: str = "WARNING", **kwargs: Any
    ) -> None:
        r"""See `ConfigurationManager.log_warning`."""
        return self.manager.log_warning(message, title=title, **kwargs)

    @Logger.log_passthrough
    def log_execute_func(
        self, fn: Callable[_P, _T], *args: _P.args, **kwargs: _P.kwargs
    ) -> _T:
        r"""See `ConfigurationManager.log_execute_func`."""
        return self.manager.log_execute_func(fn, *args, **kwargs)

    @Logger.log_passthrough
    def log_execute_command(
        self, command: Sequence[_T]
    ) -> CompletedProcess[str]:
        r"""See `ConfigurationManager.log_execute_command`."""
        return self.manager.log_execute_command(command)

    def setup(self) -> None:
        r"""Setup a `Configurable` for later configuration. By default,
        does nothing.
        """
        pass

    def configure(self) -> None:
        r"""Configure a `Configurable`, setting any options. By default, does
        nothing.
        """
        pass

    def finalize(self) -> None:
        r"""Finalize a `Configurable`. By default, does nothing."""
        pass
