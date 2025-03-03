# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from ...defaults import SMALL_SYSMEM
from ..test_stage import TestStage
from ..util import Shard, StageSpec, adjust_workers

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

    def __init__(self, config: Config, system: TestSystem) -> None:
        self._init(config, system)

    def stage_env(
        self,
        config: Config,  # noqa: ARG002
        system: TestSystem,  # noqa: ARG002
    ) -> EnvDict:
        return {}

    def shard_args(self, shard: Shard, config: Config) -> ArgList:
        return [
            "--cpus",
            "1",
            "--cpu-bind",
            str(shard),
            "--sysmem",
            str(SMALL_SYSMEM),
            "--utility",
            str(config.core.utility),
        ]

    def compute_spec(self, config: Config, system: TestSystem) -> StageSpec:
        N = len(system.cpus)
        bloat_factor = config.execution.bloat_factor

        mem_workers = system.memory // (SMALL_SYSMEM * bloat_factor)

        workers = min(N, mem_workers, 60)  # LEGION_MAX_NUM_PROCS just in case

        detail = f"{mem_workers=}"
        workers = adjust_workers(
            workers, config.execution.workers, detail=detail
        )

        # Just put each worker on its own full CPU for eager tests
        shards = [Shard([cpu.ids]) for cpu in system.cpus]
        return StageSpec(workers, shards)
