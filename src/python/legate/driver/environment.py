# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING

from ..util.shared_args import (
    AUTO_CONFIG,
    CONSENSUS,
    DISABLE_MPI,
    EXPERIMENTAL_COPY_PATH,
    FIELD_REUSE_FRACTION,
    FIELD_REUSE_FREQUENCY,
    INLINE_TASK_LAUNCH,
    IO_USE_VFD_GDS,
    MAX_EXCEPTION_SIZE,
    MIN_CPU_CHUNK,
    MIN_GPU_CHUNK,
    MIN_OMP_CHUNK,
    SHOW_CONFIG,
    SHOW_MEMORY_USAGE,
    SHOW_PROGRESS,
    SINGLE_CONTROLLER_EXECUTION,
    WARMUP_NCCL,
    WINDOW_SIZE,
)

if TYPE_CHECKING:
    from typing import Any

    from ..util.types import EnvPart
    from .config import ConfigProtocol

__all__ = ("ENV_PARTS_LEGATE",)


def _arg_helper(arg: str, value: Any) -> tuple[str, ...]:
    return () if value is None else (arg, str(value))


def env_cpus(config: ConfigProtocol) -> EnvPart:
    return _arg_helper("--cpus", config.core.cpus)


def env_gpus(config: ConfigProtocol) -> EnvPart:
    return _arg_helper("--gpus", config.core.gpus)


def env_omps(config: ConfigProtocol) -> EnvPart:
    return _arg_helper("--omps", config.core.omps)


def env_ompthreads(config: ConfigProtocol) -> EnvPart:
    return _arg_helper("--ompthreads", config.core.ompthreads)


def env_utility(config: ConfigProtocol) -> EnvPart:
    return _arg_helper("--utility", config.core.utility)


def env_sysmem(config: ConfigProtocol) -> EnvPart:
    return _arg_helper("--sysmem", config.memory.sysmem)


def env_numamem(config: ConfigProtocol) -> EnvPart:
    return _arg_helper("--numamem", config.memory.numamem)


def env_zcmem(config: ConfigProtocol) -> EnvPart:
    return _arg_helper("--zcmem", config.memory.zcmem)


def env_fbmem(config: ConfigProtocol) -> EnvPart:
    return _arg_helper("--fbmem", config.memory.fbmem)


def env_regmem(config: ConfigProtocol) -> EnvPart:
    return _arg_helper("--regmem", config.memory.regmem)


def env_max_exception_size(config: ConfigProtocol) -> EnvPart:
    return _arg_helper(
        MAX_EXCEPTION_SIZE.name, config.memory.max_exception_size
    )


def env_min_cpu_chunk(config: ConfigProtocol) -> EnvPart:
    return _arg_helper(MIN_CPU_CHUNK.name, config.memory.min_cpu_chunk)


def env_min_gpu_chunk(config: ConfigProtocol) -> EnvPart:
    return _arg_helper(MIN_GPU_CHUNK.name, config.memory.min_gpu_chunk)


def env_min_omp_chunk(config: ConfigProtocol) -> EnvPart:
    return _arg_helper(MIN_OMP_CHUNK.name, config.memory.min_omp_chunk)


def env_field_reuse_fraction(config: ConfigProtocol) -> EnvPart:
    return _arg_helper(
        FIELD_REUSE_FRACTION.name, config.memory.field_reuse_fraction
    )


def env_field_reuse_frequency(config: ConfigProtocol) -> EnvPart:
    return _arg_helper(
        FIELD_REUSE_FREQUENCY.name, config.memory.field_reuse_frequency
    )


def env_consensus(config: ConfigProtocol) -> EnvPart:
    return (CONSENSUS.name,) if config.memory.consensus else ()


def env_log_levels(config: ConfigProtocol) -> EnvPart:
    levels = config.logging.user_logging_levels
    return ("--logging", str(levels)) if levels is not None else ()


def env_logdir(config: ConfigProtocol) -> EnvPart:
    return ("--logdir", shlex.quote(str(config.logging.logdir)))


def env_log_file(config: ConfigProtocol) -> EnvPart:
    return ("--log-to-file",) if config.logging.log_to_file else ()


def env_profile(config: ConfigProtocol) -> EnvPart:
    return ("--profile",) if config.profiling.profile else ()


def env_profile_name(config: ConfigProtocol) -> EnvPart:
    return (
        ("--profile-name", config.profiling.profile_name)
        if config.profiling.profile_name
        else ()
    )


def env_provenance(config: ConfigProtocol) -> EnvPart:
    if config.profiling.provenance is None:
        return (
            ("--provenance",)
            if config.profiling.profile or config.profiling.nsys
            else ()
        )
    return ("--provenance",) if config.profiling.provenance else ()


def env_freeze_on_error(config: ConfigProtocol) -> EnvPart:
    return ("--freeze-on-error",) if config.debugging.freeze_on_error else ()


def env_auto_config(config: ConfigProtocol) -> EnvPart:
    return (AUTO_CONFIG.name,) if config.other.auto_config else ()


def env_show_config(config: ConfigProtocol) -> EnvPart:
    return (SHOW_CONFIG.name,) if config.other.show_config else ()


def env_show_memory_usage(config: ConfigProtocol) -> EnvPart:
    return (SHOW_MEMORY_USAGE.name,) if config.other.show_memory_usage else ()


def env_show_progress(config: ConfigProtocol) -> EnvPart:
    return (SHOW_PROGRESS.name,) if config.other.show_progress else ()


def env_window_size(config: ConfigProtocol) -> EnvPart:
    return _arg_helper(WINDOW_SIZE.name, config.other.window_size)


def env_warmup_nccl(config: ConfigProtocol) -> EnvPart:
    return (WARMUP_NCCL.name,) if config.other.warmup_nccl else ()


def env_disable_mpi(config: ConfigProtocol) -> EnvPart:
    return (DISABLE_MPI.name,) if config.other.disable_mpi else ()


def env_inline_task_launch(config: ConfigProtocol) -> EnvPart:
    return (
        (INLINE_TASK_LAUNCH.name,) if config.other.inline_task_launch else ()
    )


def env_single_controller_execution(config: ConfigProtocol) -> EnvPart:
    return (
        (SINGLE_CONTROLLER_EXECUTION.name,)
        if config.other.single_controller_execution
        else ()
    )


def env_io_use_vfd_gds(config: ConfigProtocol) -> EnvPart:
    return (IO_USE_VFD_GDS.name,) if config.other.io_use_vfd_gds else ()


def env_experimental_copy_path(config: ConfigProtocol) -> EnvPart:
    return (
        (EXPERIMENTAL_COPY_PATH.name,)
        if config.other.experimental_copy_path
        else ()
    )


ENV_PARTS_LEGATE = (
    env_cpus,
    env_gpus,
    env_omps,
    env_ompthreads,
    env_utility,
    env_sysmem,
    env_numamem,
    env_fbmem,
    env_zcmem,
    env_regmem,
    env_max_exception_size,
    env_min_cpu_chunk,
    env_min_gpu_chunk,
    env_min_omp_chunk,
    env_field_reuse_fraction,
    env_field_reuse_frequency,
    env_consensus,
    env_log_levels,
    env_logdir,
    env_log_file,
    env_profile,
    env_profile_name,
    env_provenance,
    env_freeze_on_error,
    env_auto_config,
    env_show_config,
    env_show_memory_usage,
    env_show_progress,
    env_window_size,
    env_warmup_nccl,
    env_disable_mpi,
    env_inline_task_launch,
    env_single_controller_execution,
    env_io_use_vfd_gds,
    env_experimental_copy_path,
)
