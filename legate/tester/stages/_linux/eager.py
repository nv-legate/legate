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

from typing import TYPE_CHECKING

from ...defaults import SMALL_SYSMEM
from ..test_stage import TestStage
from ..util import EAGER_ENV, Shard, StageSpec, adjust_workers

if TYPE_CHECKING:
    from ....util.types import ArgList, EnvDict
    from ... import FeatureType
    from ...config import Config
    from ...test_system import TestSystem


class Eager(TestStage):
    """A test stage for exercising Eager Numpy execution features.

    Parameters
    ----------
    config: Config
        Test runner configuration

    system: TestSystem
        Process execution wrapper

    """

    kind: FeatureType = "eager"

    args: ArgList = []

    def __init__(self, config: Config, system: TestSystem) -> None:
        self._init(config, system)

    def env(self, config: Config, system: TestSystem) -> EnvDict:
        env = dict(EAGER_ENV)
        return env

    def shard_args(self, shard: Shard, config: Config) -> ArgList:
        return [
            "--cpus",
            "1",
            "--cpu-bind",
            str(shard),
            "--sysmem",
            str(SMALL_SYSMEM),
        ]

    def compute_spec(self, config: Config, system: TestSystem) -> StageSpec:
        N = len(system.cpus)
        bloat_factor = config.execution.bloat_factor

        mem_workers = system.memory // (SMALL_SYSMEM * bloat_factor)

        workers = min(N, mem_workers, 60)  # LEGION_MAX_NUM_PROCS just in case

        workers = adjust_workers(workers, config.execution.workers)

        # Just put each worker on its own full CPU for eager tests
        shards = [Shard([cpu.ids]) for cpu in system.cpus]

        return StageSpec(workers, shards)
