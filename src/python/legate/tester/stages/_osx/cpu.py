# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

from typing import TYPE_CHECKING

from ..test_stage import TestStage
from ..util import UNPIN_ENV, Shard, StageSpec, adjust_workers

if TYPE_CHECKING:
    from ....util.types import ArgList, EnvDict
    from ... import FeatureType
    from ...config import Config
    from ...test_system import TestSystem


class CPU(TestStage):
    """A test stage for exercising CPU features.

    Parameters
    ----------
    config: Config
        Test runner configuration

    system: TestSystem
        Process execution wrapper

    """

    kind: FeatureType = "cpus"

    def __init__(self, config: Config, system: TestSystem) -> None:
        self._init(config, system)

    def stage_env(
        self,
        config: Config,  # noqa: ARG002
        system: TestSystem,  # noqa: ARG002
    ) -> EnvDict:
        return dict(UNPIN_ENV)

    def shard_args(self, shard: Shard, config: Config) -> ArgList:  # noqa: ARG002
        return [
            "--cpus",
            str(config.core.cpus),
            "--sysmem",
            str(config.memory.sysmem),
            "--utility",
            str(config.core.utility),
        ]

    def compute_spec(self, config: Config, system: TestSystem) -> StageSpec:
        cpus = system.cpus
        sysmem = config.memory.sysmem
        bloat_factor = config.execution.bloat_factor

        procs = (
            config.core.cpus
            + config.core.utility
            + int(config.execution.cpu_pin == "strict")
        )

        cpu_workers = len(cpus) // (procs * config.multi_node.ranks_per_node)

        mem_workers = system.memory // (sysmem * bloat_factor)

        workers = min(cpu_workers, mem_workers)

        detail = f"{cpu_workers=} {mem_workers=}"
        workers = adjust_workers(
            workers, config.execution.workers, detail=detail
        )

        # return a dummy set of shards just for the runner to iterate over
        shards = [Shard([(i,)]) for i in range(workers)]
        return StageSpec(workers, shards)
