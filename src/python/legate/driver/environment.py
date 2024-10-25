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

import shlex
from typing import TYPE_CHECKING

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


def env_log_levels(config: ConfigProtocol) -> EnvPart:
    levels = config.logging.user_logging_levels
    return ("--logging", str(levels)) if levels is not None else ()


def env_logdir(config: ConfigProtocol) -> EnvPart:
    return ("--logdir", shlex.quote(str(config.logging.logdir)))


def env_log_file(config: ConfigProtocol) -> EnvPart:
    return ("--log-to-file",) if config.logging.log_to_file else ()


def env_eager_alloc(config: ConfigProtocol) -> EnvPart:
    return _arg_helper("--eager-alloc-percentage", config.memory.eager_alloc)


def env_profile(config: ConfigProtocol) -> EnvPart:
    return ("--profile",) if config.profiling.profile else ()


def env_spy(config: ConfigProtocol) -> EnvPart:
    return ("--spy",) if config.debugging.spy else ()


def env_freeze_on_error(config: ConfigProtocol) -> EnvPart:
    return ("--freeze-on-error",) if config.debugging.freeze_on_error else ()


ENV_PARTS_LEGATE = (
    env_cpus,
    env_gpus,
    env_omps,
    env_ompthreads,
    env_utility,
    env_sysmem,
    env_numamem,
    env_fbmem,
    env_regmem,
    env_log_levels,
    env_logdir,
    env_log_file,
    env_eager_alloc,
    env_profile,
    env_spy,
    env_freeze_on_error,
)
