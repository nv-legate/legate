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

    args: ArgList = []

    def __init__(self, config: Config, system: TestSystem) -> None:
        self._init(config, system)

    def env(self, config: Config, system: TestSystem) -> EnvDict:
        return {} if config.cpu_pin == "strict" else dict(UNPIN_ENV)

    def shard_args(self, shard: Shard, config: Config) -> ArgList:
        args = [
            "--cpus",
            str(config.cpus),
        ]
        if config.cpu_pin != "none":
            args += [
                "--cpu-bind",
                str(shard),
            ]
        if config.ranks_per_node > 1:
            args += [
                "--ranks-per-node",
                str(config.ranks_per_node),
            ]
        return args

    def compute_spec(self, config: Config, system: TestSystem) -> StageSpec:
        cpus = system.cpus

        procs = config.cpus + config.utility + int(config.cpu_pin == "strict")

        workers = len(cpus) // (procs * config.ranks_per_node)

        if workers == 0:
            if config.cpu_pin == "strict":
                raise RuntimeError(
                    f"{len(cpus)} detected core(s) not enough for "
                    f"{config.ranks_per_node} rank(s) per node, each "
                    f"reserving {procs} core(s) with strict CPU pinning"
                )
            else:
                warnings.warn(
                    f"{len(cpus)} detected core(s) not enough for "
                    f"{config.ranks_per_node} rank(s) per node, each "
                    f"reserving {procs} core(s), running anyway."
                )
                all_cpus = tuple(range(len(cpus)))
                return StageSpec(1, [Shard([all_cpus])])

        workers = adjust_workers(workers, config.requested_workers)

        shards: list[Shard] = []
        for i in range(workers):
            rank_shards = []
            for j in range(config.ranks_per_node):
                shard_cpus = range(
                    (j + i * config.ranks_per_node) * procs,
                    (j + i * config.ranks_per_node + 1) * procs,
                )
                shard = chain.from_iterable(cpus[k].ids for k in shard_cpus)
                rank_shards.append(tuple(sorted(shard)))
            shards.append(Shard(rank_shards))

        return StageSpec(workers, shards)
