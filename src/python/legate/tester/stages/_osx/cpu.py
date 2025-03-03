# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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

    def __init__(self, config: Config, system: TestSystem) -> None:
        self._init(config, system)

    def stage_env(
        self,
        config: Config,  # noqa: ARG002
        system: TestSystem,  # noqa: ARG002
    ) -> EnvDict:
        return dict(UNPIN_ENV)

    @staticmethod
    def handle_cpu_pin_args(
        config: Config,
        shard: Shard,  # noqa: ARG004
    ) -> ArgList:
        if config.execution.cpu_pin != "none":
            warnings.warn(
                "CPU pinning is not supported on macOS, ignoring pinning "
                "arguments",
                stacklevel=2,
            )

        return []

    def shard_args(self, shard: Shard, config: Config) -> ArgList:
        args = [
            "--cpus",
            str(config.core.cpus),
            "--sysmem",
            str(config.memory.sysmem),
            "--utility",
            str(config.core.utility),
        ]
        args += self.handle_cpu_pin_args(config, shard)
        args += self.handle_multi_node_args(config)
        return args

    def compute_spec(self, config: Config, system: TestSystem) -> StageSpec:
        cpus = system.cpus
        ranks_per_node = config.multi_node.ranks_per_node
        sysmem = config.memory.sysmem
        bloat_factor = config.execution.bloat_factor

        procs = (
            config.core.cpus
            + config.core.utility
            + int(config.execution.cpu_pin == "strict")
        )

        cpu_workers = len(cpus) // (procs * ranks_per_node)

        mem_workers = system.memory // (sysmem * bloat_factor)

        if cpu_workers == 0:
            if config.execution.cpu_pin == "strict":
                msg = (
                    f"{len(cpus)} detected core(s) not enough for "
                    f"{ranks_per_node} rank(s) per node, each "
                    f"reserving {procs} core(s) with strict CPU pinning. "
                    "While CPU pinning is not supported in macOS, this "
                    "configuration is nevertheless unsatisfiable. If you "
                    "would like legate to launch it anyway, run with "
                    "'--cpu-pin none'."
                )
                raise RuntimeError(msg)
            if mem_workers > 0:
                warnings.warn(
                    f"{len(cpus)} detected core(s) not enough for "
                    f"{ranks_per_node} rank(s) per node, each "
                    f"reserving {procs} core(s), running anyway.",
                    stacklevel=2,
                )
                all_cpus = chain.from_iterable(cpu.ids for cpu in cpus)
                return StageSpec(1, [Shard([tuple(sorted(all_cpus))])])

        workers = min(cpu_workers, mem_workers)

        detail = f"{cpu_workers=} {mem_workers=}"
        workers = adjust_workers(
            workers, config.execution.workers, detail=detail
        )

        # return a dummy set of shards just for the runner to iterate over
        shards = [Shard([(i,)]) for i in range(workers)]
        return StageSpec(workers, shards)
