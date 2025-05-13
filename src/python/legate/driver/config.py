# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Consolidate driver configuration from command-line and environment."""

from __future__ import annotations

import shlex
import operator
from dataclasses import dataclass
from functools import cached_property, reduce
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from rich import reconfigure

from ..util.types import (
    ArgList,
    DataclassMixin,
    LauncherType,
    RunMode,
    object_to_dataclass,
)
from .args import parser

if TYPE_CHECKING:
    from argparse import Namespace

__all__ = ("Config",)


@dataclass(frozen=True)
class MultiNode(DataclassMixin):
    nodes: int
    ranks_per_node: int
    launcher: LauncherType
    launcher_extra: list[str]

    def __post_init__(self, **kw: dict[str, Any]) -> None:
        # fix up launcher_extra to automatically handle quoted strings with
        # internal whitespace, have to use __setattr__ for frozen
        # https://docs.python.org/3/library/dataclasses.html#frozen-instances
        if self.launcher_extra:
            ex: list[str] = reduce(
                operator.iadd,
                (shlex.split(x) for x in self.launcher_extra),
                [],
            )
            object.__setattr__(self, "launcher_extra", ex)

    @property
    def ranks(self) -> int:
        return self.nodes * self.ranks_per_node


@dataclass(frozen=True)
class Binding(DataclassMixin):
    cpu_bind: str | None
    mem_bind: str | None
    gpu_bind: str | None
    nic_bind: str | None


@dataclass(frozen=True)
class Core(DataclassMixin):
    cpus: int | None
    gpus: int | None
    omps: int | None
    ompthreads: int | None
    utility: int | None

    # compat alias for old field name
    @property
    def openmp(self) -> int | None:
        return self.omps


@dataclass(frozen=True)
class Memory(DataclassMixin):
    sysmem: int | None
    numamem: int | None
    fbmem: int | None
    zcmem: int | None
    regmem: int | None


@dataclass(frozen=True)
class Profiling(DataclassMixin):
    profile: bool
    cprofile: bool
    nvprof: bool
    nsys: bool
    nsys_extra: list[str]

    def __post_init__(self, **kw: dict[str, Any]) -> None:
        # fix up nsys_extra to automatically handle quoted strings with
        # internal whitespace, have to use __setattr__ for frozen
        # https://docs.python.org/3/library/dataclasses.html#frozen-instances
        if self.nsys_extra:
            ex: list[str] = reduce(
                operator.iadd, (shlex.split(x) for x in self.nsys_extra), []
            )
            object.__setattr__(self, "nsys_extra", ex)


@dataclass(frozen=True)
class Logging(DataclassMixin):
    def __post_init__(self, **kw: dict[str, Any]) -> None:
        # fix up logdir to be a real path, have to use __setattr__ for frozen
        # https://docs.python.org/3/library/dataclasses.html#frozen-instances
        if self.logdir:
            object.__setattr__(self, "logdir", Path(self.logdir))

    user_logging_levels: str | None
    logdir: Path
    log_to_file: bool


@dataclass(frozen=True)
class Debugging(DataclassMixin):
    gdb: bool
    cuda_gdb: bool
    memcheck: bool
    valgrind: bool
    freeze_on_error: bool
    gasnet_trace: bool


@dataclass(frozen=True)
class Info(DataclassMixin):
    verbose: bool
    bind_detail: bool


@dataclass(frozen=True)
class Other(DataclassMixin):
    timing: bool
    wrapper: list[str]
    wrapper_inner: list[str]
    module: list[str] | None
    dry_run: bool
    color: bool


class ConfigProtocol(Protocol):
    _args: Namespace

    argv: ArgList

    user_program: str | None
    user_opts: tuple[str, ...]
    multi_node: MultiNode
    binding: Binding
    core: Core
    memory: Memory
    profiling: Profiling
    logging: Logging
    debugging: Debugging
    info: Info
    other: Other

    @cached_property
    def console(self) -> bool: ...

    @cached_property
    def run_mode(self) -> RunMode: ...


class Config:
    """A centralized configuration object that provides the information
    needed by the Legate driver in order to run.

    Parameters
    ----------
    argv : ArgList
        command-line arguments to use when building the configuration

    """

    def __init__(self, argv: ArgList) -> None:
        self.argv = argv

        args = parser.parse_args(self.argv[1:])

        # only saving this for help with testing
        self._args = args

        self.user_program = args.command[0] if args.command else None
        self.user_opts = tuple(args.command[1:]) if self.user_program else ()
        self._user_run_mode = args.run_mode

        self.multi_node = object_to_dataclass(args, MultiNode)
        self.binding = object_to_dataclass(args, Binding)
        self.core = object_to_dataclass(args, Core)
        self.memory = object_to_dataclass(args, Memory)
        self.profiling = object_to_dataclass(args, Profiling)
        self.logging = object_to_dataclass(args, Logging)
        self.debugging = object_to_dataclass(args, Debugging)
        self.info = object_to_dataclass(args, Info)
        self.other = object_to_dataclass(args, Other)

        if self.run_mode == "exec":
            if self.user_program is None:
                msg = "'exec' run mode requires a program to execute"
                raise RuntimeError(msg)
            if self.other.module is not None:
                msg = "'exec' run mode cannot be used with --module"
                raise RuntimeError(msg)

        color_system = "auto" if self.other.color else None
        reconfigure(soft_wrap=True, color_system=color_system)

    @cached_property
    def console(self) -> bool:
        """Whether we are starting Legate as an interactive console."""
        return (
            self.user_program is None
            and not self.other.module
            and self.run_mode == "python"
        )

    @cached_property
    def run_mode(self) -> RunMode:  # noqa: D102
        # honor any explicit user configuration
        if self._user_run_mode is not None:
            return self._user_run_mode

        # no user program, just run python
        if self.user_program is None:
            return "python"

        # --module specified means run with python
        if self.other.module is not None:
            return "python"

        # otherwise assume .py means run with python
        if self.user_program.endswith(".py"):
            return "python"

        return "exec"
