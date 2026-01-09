# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Consolidate driver configuration from command-line and environment."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from legate.util.types import RunMode

from legate.driver.config import (
    Binding,
    Core,
    Debugging,
    Info,
    Logging,
    Memory,
    MultiNode,
    Other,
    Profiling,
)
from legate.jupyter.args import parser
from legate.util.types import ArgList, DataclassMixin, object_to_dataclass

__all__ = ("Config",)


@dataclass(frozen=True)
class Kernel(DataclassMixin):
    user: bool
    prefix: str | None
    spec_name: str
    display_name: str


class Config:
    """A Jupyter-specific configuration object that provides the information
    needed by the Legate driver in order to run.

    Parameters
    ----------
    argv : ArgList
        command-line arguments to use when building the configuration

    """

    def __init__(self, argv: ArgList) -> None:
        self.argv = argv

        args = parser.parse_args(self.argv[1:])

        # only saving these for help with testing
        self._args = args

        if args.display_name is None:
            args.display_name = args.spec_name

        self.kernel = object_to_dataclass(args, Kernel)
        self.verbose = args.verbose

        # these are the values we leave configurable for the kernel
        self.multi_node = object_to_dataclass(args, MultiNode)
        self.core = object_to_dataclass(args, Core)
        self.memory = object_to_dataclass(args, Memory)

        # need to override explicitly since there is no user program or args
        self._user_run_mode = "python"

        # turn everything else off
        self.user_program: str | None = None
        self.user_opts: tuple[str, ...] = ()
        self.binding = Binding(
            cpu_bind=None, mem_bind=None, gpu_bind=None, nic_bind=None
        )
        self.profiling = Profiling(
            profile=False,
            profile_name=None,
            provenance=None,
            cprofile=False,
            nvprof=False,
            nsys=False,
            nsys_extra=[],
        )
        self.logging = Logging(
            user_logging_levels=None, logdir=Path(), log_to_file=False
        )
        self.debugging = Debugging(
            gdb=False,
            cuda_gdb=False,
            memcheck=False,
            valgrind=False,
            freeze_on_error=False,
            gasnet_trace=False,
        )
        self.info = Info(verbose=self.verbose > 0, bind_detail=False)
        self.other = Other(
            auto_config=bool(args.auto_config),
            show_config=bool(args.show_config),
            show_memory_usage=bool(args.show_memory_usage),
            show_progress=bool(args.show_progress),
            timing=False,
            wrapper=[],
            wrapper_inner=[],
            module=None,
            dry_run=False,
            color=False,
            window_size=args.window_size,
            warmup_nccl=args.warmup_nccl,
            disable_mpi=args.disable_mpi,
            inline_task_launch=args.inline_task_launch,
            single_controller_execution=args.single_controller_execution,
            io_use_vfd_gds=args.io_use_vfd_gds,
            experimental_copy_path=args.experimental_copy_path,
        )

    @cached_property
    def run_mode(self) -> RunMode:  # noqa: D102
        return "python"

    @cached_property
    def console(self) -> bool:  # noqa: D102
        return True
