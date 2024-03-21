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

import warnings
from itertools import chain
from typing import TYPE_CHECKING

from ...defaults import SMALL_SYSMEM
from ..test_stage import TestStage
from ..util import UNPIN_ENV, Shard, StageSpec, adjust_workers

if TYPE_CHECKING:
    from ....util.types import ArgList, EnvDict
    from ... import FeatureType
    from ...config import Config
    from ...test_system import TestSystem


class OMP(TestStage):
    """A test stage for exercising OpenMP features.

    Parameters
    ----------
    config: Config
        Test runner configuration

    system: TestSystem
        Process execution wrapper

    """

    kind: FeatureType = "openmp"

    args: ArgList = []

    def __init__(self, config: Config, system: TestSystem) -> None:
        self._init(config, system)

    def env(self, config: Config, system: TestSystem) -> EnvDict:
        return {} if config.execution.cpu_pin == "strict" else dict(UNPIN_ENV)

    def shard_args(self, shard: Shard, config: Config) -> ArgList:
        args = [
            "--omps",
            str(config.core.omps),
            "--ompthreads",
            str(config.core.ompthreads),
            "--numamem",
            str(config.memory.numamem),
            "--sysmem",
            str(SMALL_SYSMEM),
        ]
        args += self.handle_cpu_pin_args(config, shard)
        args += self.handle_multi_node_args(config)
        return args

    def compute_spec(self, config: Config, system: TestSystem) -> StageSpec:
        cpus = system.cpus
        omps, threads = config.core.omps, config.core.ompthreads
        ranks_per_node = config.multi_node.ranks_per_node
        numamem = config.memory.numamem
        bloat_factor = config.execution.bloat_factor

        procs = (
            omps * threads
            + config.core.utility
            + int(config.execution.cpu_pin == "strict")
        )

        omp_workers = len(cpus) // (procs * ranks_per_node)

        mem_per_test = (SMALL_SYSMEM + omps * numamem) * bloat_factor

        mem_workers = system.memory // mem_per_test

        workers = min(omp_workers, mem_workers)

        if workers == 0:
            if config.execution.cpu_pin == "strict":
                raise RuntimeError(
                    f"{len(cpus)} detected core(s) not enough for "
                    f"{ranks_per_node} rank(s) per node, each "
                    f"reserving {procs} core(s) with strict CPU pinning"
                )
            else:
                warnings.warn(
                    f"{len(cpus)} detected core(s) not enough for "
                    f"{ranks_per_node} rank(s) per node, each "
                    f"reserving {procs} core(s), running anyway."
                )
                all_cpus = chain.from_iterable(cpu.ids for cpu in cpus)
                return StageSpec(1, [Shard([tuple(sorted(all_cpus))])])

        workers = adjust_workers(workers, config.execution.workers)

        shards: list[Shard] = []
        for i in range(workers):
            rank_shards = []
            for j in range(ranks_per_node):
                shard_cpus = range(
                    (j + i * ranks_per_node) * procs,
                    (j + i * ranks_per_node + 1) * procs,
                )
                shard = chain.from_iterable(cpus[k].ids for k in shard_cpus)
                rank_shards.append(tuple(sorted(shard)))
            shards.append(Shard(rank_shards))

        return StageSpec(workers, shards)
