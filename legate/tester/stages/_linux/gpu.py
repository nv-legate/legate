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

    args: ArgList = []

    def __init__(self, config: Config, system: TestSystem) -> None:
        self._init(config, system)

    def env(self, config: Config, system: TestSystem) -> EnvDict:
        return {}

    def delay(self, shard: Shard, config: Config, system: TestSystem) -> None:
        time.sleep(config.execution.gpu_delay / 1000)

    def shard_args(self, shard: Shard, config: Config) -> ArgList:
        args = [
            "--fbmem",
            str(config.memory.fbmem),
            "--gpus",
            str(sum(len(r) for r in shard.ranks) // len(shard.ranks)),
            "--gpu-bind",
            str(shard),
            "--sysmem",
            str(SMALL_SYSMEM),
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

        workers = adjust_workers(
            min(gpu_workers, mem_workers), config.execution.workers
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
