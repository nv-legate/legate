# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from ..driver import defaults
from .args import ArgSpec, Argument

if TYPE_CHECKING:
    from .types import LauncherType

__all__ = (
    "CPUS",
    "FBMEM",
    "GPUS",
    "LAUNCHER",
    "LAUNCHERS",
    "LAUNCHER_EXTRA",
    "NODES",
    "NUMAMEM",
    "OMPS",
    "OMPTHREADS",
    "RANKS_PER_NODE",
    "REGMEM",
    "SYSMEM",
    "UTILITY",
    "ZCMEM",
)

LAUNCHERS: tuple[LauncherType, ...] = (
    "mpirun",
    "jsrun",
    "srun",
    "dask",
    "none",
)

NODES = Argument(
    "--nodes",
    ArgSpec(
        type=int,
        default=defaults.LEGATE_NODES,
        dest="nodes",
        help="Number of nodes to use. "
        "[legate-only, not supported with standard Python invocation]",
    ),
)


RANKS_PER_NODE = Argument(
    "--ranks-per-node",
    ArgSpec(
        type=int,
        default=defaults.LEGATE_RANKS_PER_NODE,
        dest="ranks_per_node",
        help="Number of ranks (processes running copies of the program) to "
        "launch per node. 1 rank per node will typically result in the best "
        "performance. "
        "[legate-only, not supported with standard Python invocation]",
    ),
)


LAUNCHER = Argument(
    "--launcher",
    ArgSpec(
        dest="launcher",
        choices=LAUNCHERS,
        default="none",
        help='launcher program to use (set to "none" for local runs, or if '
        "the launch has already happened by the time legate is invoked), "
        "[legate-only, not supported with standard Python invocation]",
    ),
)


LAUNCHER_EXTRA = Argument(
    "--launcher-extra",
    ArgSpec(
        dest="launcher_extra",
        action="append",
        default=[],
        required=False,
        help="additional argument to pass to the launcher (can appear more "
        "than once). Multiple arguments may be provided together in a quoted "
        "string (arguments with spaces inside must be additionally quoted), "
        "[legate-only, not supported with standard Python invocation]",
    ),
)


CPUS = Argument(
    "--cpus",
    ArgSpec(
        type=int,
        default=None,
        dest="cpus",
        help="Number of standalone CPU cores to reserve per rank, must be >=0",
    ),
)


GPUS = Argument(
    "--gpus",
    ArgSpec(
        type=int,
        default=None,
        dest="gpus",
        help="Number of GPUs to reserve per rank, must be >=0",
    ),
)


OMPS = Argument(
    "--omps",
    ArgSpec(
        type=int,
        default=None,
        dest="omps",
        help="Number of OpenMP groups to use per rank, must be >=0",
    ),
)


OMPTHREADS = Argument(
    "--ompthreads",
    ArgSpec(
        type=int,
        default=None,
        dest="ompthreads",
        help="Number of threads / reserved CPU cores per OpenMP group, must "
        "be >=0",
    ),
)


UTILITY = Argument(
    "--utility",
    ArgSpec(
        type=int,
        default=None,
        dest="utility",
        help="Number of threads to use per rank for runtime meta-work, must "
        "be >=0",
    ),
)


SYSMEM = Argument(
    "--sysmem",
    ArgSpec(
        type=int,
        default=None,
        dest="sysmem",
        help="Size (in MiB) of DRAM memory to reserve per rank",
    ),
)


NUMAMEM = Argument(
    "--numamem",
    ArgSpec(
        type=int,
        default=None,
        dest="numamem",
        help="Size (in MiB) of NUMA-specific DRAM memory to reserve per NUMA "
        "domain per rank",
    ),
)


FBMEM = Argument(
    "--fbmem",
    ArgSpec(
        type=int,
        default=None,
        dest="fbmem",
        help='Size (in MiB) of GPU (or "framebuffer") memory to reserve per '
        "GPU",
    ),
)


ZCMEM = Argument(
    "--zcmem",
    ArgSpec(
        type=int,
        default=None,
        dest="zcmem",
        help='Size (in MiB) of GPU-registered (or "zero-copy") DRAM memory '
        "to reserve per GPU",
    ),
)


REGMEM = Argument(
    "--regmem",
    ArgSpec(
        type=int,
        default=None,
        dest="regmem",
        help="Size (in MiB) of NIC-registered DRAM memory to reserve per rank",
    ),
)
