# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
        return dict(UNPIN_ENV)

    def shard_args(self, shard: Shard, config: Config) -> ArgList:
        return [
            "--omps",
            str(config.omps),
            "--ompthreads",
            str(config.ompthreads),
        ]

    def compute_spec(self, config: Config, system: TestSystem) -> StageSpec:
        omps, threads = config.omps, config.ompthreads
        procs = omps * threads + config.utility
        workers = adjust_workers(
            len(system.cpus) // procs, config.requested_workers
        )

        # return a dummy set of shards just for the runner to iterate over
        shards = [Shard([(i,)]) for i in range(workers)]
        return StageSpec(workers, shards)
