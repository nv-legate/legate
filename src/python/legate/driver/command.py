# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING

from rich import print as rich_print

from .. import install_info
from ..util.ui import warn

if TYPE_CHECKING:
    from ..util.system import System
    from ..util.types import CommandPart
    from .config import ConfigProtocol
    from .launcher import Launcher

__all__ = ("CMD_PARTS_EXEC", "CMD_PARTS_PYTHON")


# this will be replaced by legate-bind.sh with the actual computed rank at
# runtime
LEGATE_GLOBAL_RANK_SUBSTITUTION = "%%LEGATE_GLOBAL_RANK%%"


def cmd_bind(
    config: ConfigProtocol, system: System, launcher: Launcher
) -> CommandPart:
    ranks = config.multi_node.ranks

    if ranks > 1 and len(install_info.networks) == 0:
        msg = (
            "multi-rank run was requested, but Legate was not built with "
            "networking support"
        )
        raise RuntimeError(msg)

    if launcher.kind == "none":
        bind_launcher_arg = "local" if ranks == 1 else "auto"
    else:
        bind_launcher_arg = launcher.kind

    opts: CommandPart = (
        str(system.legate_paths.bind_sh_path),
        "--launcher",
        bind_launcher_arg,
    )

    ranks_per_node = config.multi_node.ranks_per_node

    errmsg = "Number of groups in --{name}-bind not equal to --ranks-per-node"

    def check_bind_ranks(name: str, binding: str) -> None:
        if len(binding.split("/")) != ranks_per_node:
            raise RuntimeError(errmsg.format(name=name))

    bindings = (
        ("cpu", config.binding.cpu_bind),
        ("gpu", config.binding.gpu_bind),
        ("mem", config.binding.mem_bind),
        ("nic", config.binding.nic_bind),
    )
    for name, binding in bindings:
        if binding is not None:
            check_bind_ranks(name, binding)
            opts += (f"--{name}s", binding)

    if config.info.bind_detail:
        opts += ("--debug",)

    return (*opts, "--")


def cmd_gdb(
    config: ConfigProtocol,
    system: System,
    launcher: Launcher,  # noqa: ARG001
) -> CommandPart:
    if not config.debugging.gdb:
        return ()

    if config.multi_node.ranks > 1:
        rich_print(warn("Legate does not support gdb for multi-rank runs"))
        return ()

    return ("lldb", "--") if system.os == "Darwin" else ("gdb", "--args")


def cmd_cuda_gdb(
    config: ConfigProtocol,
    system: System,  # noqa: ARG001
    launcher: Launcher,  # noqa: ARG001
) -> CommandPart:
    if not config.debugging.cuda_gdb:
        return ()

    if config.multi_node.ranks > 1:
        rich_print(
            warn("Legate does not support cuda-gdb for multi-rank runs")
        )
        return ()

    return ("cuda-gdb", "--args")


def cmd_nvprof(
    config: ConfigProtocol,
    system: System,  # noqa: ARG001
    launcher: Launcher,  # noqa: ARG001
) -> CommandPart:
    if not config.profiling.nvprof:
        return ()

    log_path = str(
        config.logging.logdir
        / f"legate_{LEGATE_GLOBAL_RANK_SUBSTITUTION}.nvvp"
    )

    return ("nvprof", "-o", log_path)


def cmd_nsys(
    config: ConfigProtocol,
    system: System,  # noqa: ARG001
    launcher: Launcher,  # noqa: ARG001
) -> CommandPart:
    if not config.profiling.nsys:
        return ()

    log_path = str(
        config.logging.logdir / f"legate_{LEGATE_GLOBAL_RANK_SUBSTITUTION}"
    )
    extra = config.profiling.nsys_extra

    return ("nsys", "profile", "-o", log_path, *tuple(extra))


def cmd_valgrind(
    config: ConfigProtocol,
    system: System,  # noqa: ARG001
    launcher: Launcher,  # noqa: ARG001
) -> CommandPart:
    valgrind = config.debugging.valgrind

    return () if not valgrind else ("valgrind",)


def cmd_memcheck(
    config: ConfigProtocol,
    system: System,  # noqa: ARG001
    launcher: Launcher,  # noqa: ARG001
) -> CommandPart:
    memcheck = config.debugging.memcheck

    return () if not memcheck else ("compute-sanitizer",)


def cmd_module(
    config: ConfigProtocol,
    system: System,  # noqa: ARG001
    launcher: Launcher,  # noqa: ARG001
) -> CommandPart:
    module = config.other.module
    cprofile = config.profiling.cprofile

    if cprofile and module is not None:
        msg = "Only one of --module or --cprofile may be used"
        raise ValueError(msg)

    if module is not None:
        return ("-m", *module)

    if cprofile:
        log_path = str(
            config.logging.logdir
            / f"legate_{LEGATE_GLOBAL_RANK_SUBSTITUTION}.cprof"
        )
        return ("-m", "cProfile", "-o", log_path)

    return ()


def cmd_wrapper(
    config: ConfigProtocol,
    system: System,  # noqa: ARG001
    launcher: Launcher,  # noqa: ARG001
) -> CommandPart:
    if not config.other.wrapper:
        return ()

    parts = []

    for wrapper in config.other.wrapper:
        parts.extend(shlex.split(wrapper))

    return tuple(parts)


def cmd_wrapper_inner(
    config: ConfigProtocol,
    system: System,  # noqa: ARG001
    launcher: Launcher,  # noqa: ARG001
) -> CommandPart:
    if not config.other.wrapper_inner:
        return ()

    parts = []

    for wrapper in config.other.wrapper_inner:
        parts.extend(shlex.split(wrapper))

    return tuple(parts)


def cmd_user_program(
    config: ConfigProtocol,
    system: System,  # noqa: ARG001
    launcher: Launcher,  # noqa: ARG001
) -> CommandPart:
    return () if config.user_program is None else (config.user_program,)


def cmd_user_opts(
    config: ConfigProtocol,
    system: System,  # noqa: ARG001
    launcher: Launcher,  # noqa: ARG001
) -> CommandPart:
    return config.user_opts


def cmd_python(
    config: ConfigProtocol,  # noqa: ARG001
    system: System,  # noqa: ARG001
    launcher: Launcher,  # noqa: ARG001
) -> CommandPart:
    return ("python",)


_CMD_PARTS_PRE = (
    cmd_bind,
    # Add any user supplied (outer) wrappers
    cmd_wrapper,
    cmd_gdb,
    cmd_cuda_gdb,
    cmd_nvprof,
    cmd_nsys,
    # Add memcheck right before the binary
    cmd_memcheck,
    # Add valgrind right before the binary
    cmd_valgrind,
    # Add any user supplied inner wrappers
    cmd_wrapper_inner,
)

CMD_PARTS_PYTHON = (
    *_CMD_PARTS_PRE,
    # Executable name that will get stripped by the runtime
    cmd_python,
    cmd_module,
    # User script
    cmd_user_program,
    # Append user flags so they can override whatever we provided
    cmd_user_opts,
)

CMD_PARTS_EXEC = (
    *_CMD_PARTS_PRE,
    # Now we're ready to build the actual command to run
    cmd_user_program,
    # Append user flags so they can override whatever we provided
    cmd_user_opts,
)
