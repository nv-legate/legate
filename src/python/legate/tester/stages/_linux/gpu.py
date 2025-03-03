# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from ...defaults import SMALL_SYSMEM
from ..test_stage import TestStage
from ..util import Shard, StageSpec, adjust_workers

if TYPE_CHECKING:
    from ....util.types import ArgList, EnvDict
    from ... import FeatureType
    from ...config import Config
    from ...test_system import TestSystem


class GPU(TestStage):
    """A test stage for exercising GPU features.

    Parameters
    ----------
    config: Config
        Test runner configuration

    system: TestSystem
        Process execution wrapper

    """

    kind: FeatureType = "cuda"

    def __init__(self, config: Config, system: TestSystem) -> None:
        self._init(config, system)
        self._gpus = system.gpus

    def stage_env(
        self,
        config: Config,  # noqa: ARG002
        system: TestSystem,  # noqa: ARG002
    ) -> EnvDict:
        return {}

    def delay(
        self,
        shard: Shard,  # noqa: ARG002
        config: Config,
        system: TestSystem,  # noqa: ARG002
    ) -> None:
        time.sleep(config.execution.gpu_delay / 1000)

    def shard_args(self, shard: Shard, config: Config) -> ArgList:
        shard_gpu_ids = Shard(
            ranks=[
                tuple(self._gpus[r].id for r in rank) for rank in shard.ranks
            ]
        )

        args = [
            "--fbmem",
            str(config.memory.fbmem),
            "--gpus",
            str(sum(len(r) for r in shard.ranks) // len(shard.ranks)),
            "--gpu-bind",
            str(shard_gpu_ids),
            "--sysmem",
            str(SMALL_SYSMEM),
            "--cpus",
            "1",
            "--utility",
            str(config.core.utility),
        ]
        args += self.handle_multi_node_args(config)
        return args

    def compute_spec(self, config: Config, system: TestSystem) -> StageSpec:
        N = len(system.gpus)
        ranks_per_node = config.multi_node.ranks_per_node
        degree = N // (config.core.gpus * ranks_per_node)
        fbmem = config.memory.fbmem
        bloat_factor = config.execution.bloat_factor

        fbsize = min(gpu.total for gpu in system.gpus) / (1 << 20)  # MB
        oversub_factor = int(fbsize // (fbmem * bloat_factor))

        gpu_workers = degree * oversub_factor

        mem_workers = system.memory // (SMALL_SYSMEM * bloat_factor)

        workers = min(gpu_workers, mem_workers)

        detail = f"{fbsize=} {oversub_factor=} {gpu_workers=} {mem_workers=}"
        workers = adjust_workers(
            workers, config.execution.workers, detail=detail
        )

        shards: list[Shard] = []
        for i in range(degree):
            rank_shards = []
            for j in range(ranks_per_node):
                shard_gpus = range(
                    (j + i * ranks_per_node) * config.core.gpus,
                    (j + i * ranks_per_node + 1) * config.core.gpus,
                )
                shard = tuple(shard_gpus)
                rank_shards.append(shard)
            shards.append(Shard(rank_shards))

        shard_factor = workers if ranks_per_node == 1 else oversub_factor

        return StageSpec(workers, shards * shard_factor)
