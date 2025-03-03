# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from ..test_stage import TestStage
from ..util import UNPIN_ENV

if TYPE_CHECKING:
    from ....util.types import ArgList, EnvDict
    from ... import FeatureType
    from ...config import Config
    from ...test_system import TestSystem
    from ..util import Shard, StageSpec


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

    def __init__(
        self,
        config: Config,  # noqa: ARG002
        system: TestSystem,  # noqa: ARG002
    ) -> None:
        msg = "GPU test are not supported on OSX"
        raise RuntimeError(msg)

    def stage_env(
        self,
        config: Config,  # noqa: ARG002
        system: TestSystem,  # noqa: ARG002
    ) -> EnvDict:
        return dict(UNPIN_ENV)

    def delay(
        self,
        shard: Shard,  # noqa: ARG002
        config: Config,
        system: TestSystem,  # noqa: ARG002
    ) -> None:
        time.sleep(config.execution.gpu_delay / 1000)

    def shard_args(self, shard: Shard, config: Config) -> ArgList:
        raise NotImplementedError

    def compute_spec(self, config: Config, system: TestSystem) -> StageSpec:
        raise NotImplementedError
