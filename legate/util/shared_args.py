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

from ..driver import defaults
from .args import ArgSpec, Argument
from .types import LauncherType

__all__ = (
    "CPUS",
    "FBMEM",
    "GPUS",
    "LAUNCHER_EXTRA",
    "LAUNCHER",
    "LAUNCHERS",
    "NOCR",
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

LAUNCHERS: tuple[LauncherType, ...] = ("mpirun", "jsrun", "srun", "none")

NODES = Argument(
    "--nodes",
    ArgSpec(
        type=int,
        default=defaults.LEGATE_NODES,
        dest="nodes",
        help="Number of nodes to use",
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
        "performance.",
    ),
)


NOCR = Argument(
    "--no-replicate",
    ArgSpec(
        dest="not_control_replicable",
        action="store_true",
        required=False,
        help="Execute this program without control replication.  Most of the "
        "time, this is not recommended.  This option should be used for "
        "debugging.  The -lg:safe_ctrlrepl Legion option may be helpful "
        "with discovering issues with replicated control.",
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
        default=defaults.LEGATE_CPUS,
        dest="cpus",
        help="Number of CPUs to use per rank",
    ),
)

GPUS = Argument(
    "--gpus",
    ArgSpec(
        type=int,
        default=defaults.LEGATE_GPUS,
        dest="gpus",
        help="Number of GPUs to use per rank",
    ),
)

OMPS = Argument(
    "--omps",
    ArgSpec(
        type=int,
        default=defaults.LEGATE_OMP_PROCS,
        dest="openmp",
        help="Number of OpenMP groups to use per rank",
    ),
)


OMPTHREADS = Argument(
    "--ompthreads",
    ArgSpec(
        type=int,
        default=defaults.LEGATE_OMP_THREADS,
        dest="ompthreads",
        help="Number of threads per OpenMP group",
    ),
)

UTILITY = Argument(
    "--utility",
    ArgSpec(
        type=int,
        default=defaults.LEGATE_UTILITY_CORES,
        dest="utility",
        help="Number of Utility processors per rank to request for meta-work",
    ),
)

SYSMEM = Argument(
    "--sysmem",
    ArgSpec(
        type=int,
        default=defaults.LEGATE_SYSMEM,
        dest="sysmem",
        help="Amount of DRAM memory per rank (in MBs)",
    ),
)


NUMAMEM = Argument(
    "--numamem",
    ArgSpec(
        type=int,
        default=defaults.LEGATE_NUMAMEM,
        dest="numamem",
        help="Amount of DRAM memory per NUMA domain per rank (in MBs)",
    ),
)


FBMEM = Argument(
    "--fbmem",
    ArgSpec(
        type=int,
        default=defaults.LEGATE_FBMEM,
        dest="fbmem",
        help="Amount of framebuffer memory per GPU (in MBs)",
    ),
)


ZCMEM = Argument(
    "--zcmem",
    ArgSpec(
        type=int,
        default=defaults.LEGATE_ZCMEM,
        dest="zcmem",
        help="Amount of zero-copy memory per rank (in MBs)",
    ),
)


REGMEM = Argument(
    "--regmem",
    ArgSpec(
        type=int,
        default=defaults.LEGATE_REGMEM,
        dest="regmem",
        help="Amount of registered CPU-side pinned memory per rank (in MBs)",
    ),
)
