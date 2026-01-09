# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from . import defaults
from .args import ArgSpec, Argument

if TYPE_CHECKING:
    from .types import LauncherType

__all__ = (
    "AUTO_CONFIG",
    "CONSENSUS",
    "CPUS",
    "DISABLE_MPI",
    "EXPERIMENTAL_COPY_PATH",
    "FBMEM",
    "FIELD_REUSE_FRACTION",
    "FIELD_REUSE_FREQUENCY",
    "GPUS",
    "INLINE_TASK_LAUNCH",
    "IO_USE_VFD_GDS",
    "LAUNCHER",
    "LAUNCHERS",
    "LAUNCHER_EXTRA",
    "MAX_EXCEPTION_SIZE",
    "MIN_CPU_CHUNK",
    "MIN_GPU_CHUNK",
    "MIN_OMP_CHUNK",
    "NODES",
    "NUMAMEM",
    "OMPS",
    "OMPTHREADS",
    "RANKS_PER_NODE",
    "REGMEM",
    "SHOW_CONFIG",
    "SHOW_MEMORY_USAGE",
    "SHOW_PROGRESS",
    "SINGLE_CONTROLLER_EXECUTION",
    "SYSMEM",
    "UTILITY",
    "WARMUP_NCCL",
    "WINDOW_SIZE",
    "ZCMEM",
)

LAUNCHERS: tuple[LauncherType, ...] = (
    "mpirun",
    "jsrun",
    "aprun",
    "srun",
    "dask",
    "none",
)

AUTO_CONFIG = Argument(
    "--auto-config",
    ArgSpec(
        dest="auto_config",
        action="store_true",
        required=False,
        help="Set LEGATE_CONFIG='--help' for information on this option.",
    ),
)

SHOW_CONFIG = Argument(
    "--show-config",
    ArgSpec(
        dest="show_config",
        action="store_true",
        required=False,
        help="Set LEGATE_CONFIG='--help' for information on this option.",
    ),
)

SHOW_MEMORY_USAGE = Argument(
    "--show-memory-usage",
    ArgSpec(
        dest="show_memory_usage",
        action="store_true",
        required=False,
        help="Set LEGATE_CONFIG='--help' for information on this option.",
    ),
)

SHOW_PROGRESS = Argument(
    "--show-progress",
    ArgSpec(
        dest="show_progress",
        action="store_true",
        required=False,
        help="Set LEGATE_CONFIG='--help' for information on this option.",
    ),
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

MAX_EXCEPTION_SIZE = Argument(
    "--max-exception-size",
    ArgSpec(
        dest="max_exception_size",
        required=False,
        type=int,
        help="Set LEGATE_CONFIG='--help' for information on this option.",
    ),
)

MIN_CPU_CHUNK = Argument(
    "--min-cpu-chunk",
    ArgSpec(
        dest="min_cpu_chunk",
        required=False,
        type=int,
        help="Set LEGATE_CONFIG='--help' for information on this option.",
    ),
)

MIN_GPU_CHUNK = Argument(
    "--min-gpu-chunk",
    ArgSpec(
        dest="min_gpu_chunk",
        required=False,
        type=int,
        help="Set LEGATE_CONFIG='--help' for information on this option.",
    ),
)

MIN_OMP_CHUNK = Argument(
    "--min-omp-chunk",
    ArgSpec(
        dest="min_omp_chunk",
        required=False,
        type=int,
        help="Set LEGATE_CONFIG='--help' for information on this option.",
    ),
)

FIELD_REUSE_FRACTION = Argument(
    "--field-reuse-fraction",
    ArgSpec(
        dest="field_reuse_fraction",
        required=False,
        type=int,
        help="Set LEGATE_CONFIG='--help' for information on this option.",
    ),
)

FIELD_REUSE_FREQUENCY = Argument(
    "--field-reuse-frequency",
    ArgSpec(
        dest="field_reuse_frequency",
        required=False,
        type=int,
        help="Set LEGATE_CONFIG='--help' for information on this option.",
    ),
)

CONSENSUS = Argument(
    "--consensus",
    ArgSpec(
        dest="consensus",
        action="store_true",
        help="Set LEGATE_CONFIG='--help' for information on this option.",
    ),
)

WINDOW_SIZE = Argument(
    "--window-size",
    ArgSpec(
        dest="window_size",
        type=int,
        help="Set LEGATE_CONFIG='--help' for information on this option.",
    ),
)

WARMUP_NCCL = Argument(
    "--warmup-nccl",
    ArgSpec(
        dest="warmup_nccl",
        action="store_true",
        help="Set LEGATE_CONFIG='--help' for information on this option.",
    ),
)

DISABLE_MPI = Argument(
    "--disable-mpi",
    ArgSpec(
        dest="disable_mpi",
        action="store_true",
        help="Set LEGATE_CONFIG='--help' for information on this option.",
    ),
)

INLINE_TASK_LAUNCH = Argument(
    "--inline-task-launch",
    ArgSpec(
        dest="inline_task_launch",
        action="store_true",
        help="Set LEGATE_CONFIG='--help' for information on this option.",
    ),
)

SINGLE_CONTROLLER_EXECUTION = Argument(
    "--single-controller-execution",
    ArgSpec(
        dest="single_controller_execution",
        action="store_true",
        help="Set LEGATE_CONFIG='--help' for information on this option.",
    ),
)

IO_USE_VFD_GDS = Argument(
    "--io-use-vfd-gds",
    ArgSpec(
        dest="io_use_vfd_gds",
        action="store_true",
        help="Set LEGATE_CONFIG='--help' for information on this option.",
    ),
)

EXPERIMENTAL_COPY_PATH = Argument(
    "--experimental-copy-path",
    ArgSpec(
        dest="experimental_copy_path",
        action="store_true",
        help="Set LEGATE_CONFIG='--help' for information on this option.",
    ),
)
