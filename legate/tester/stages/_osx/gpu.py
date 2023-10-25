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

    args: ArgList = []

    def __init__(self, config: Config, system: TestSystem) -> None:
        raise RuntimeError("GPU test are not supported on OSX")

    def env(self, config: Config, system: TestSystem) -> EnvDict:
        return dict(UNPIN_ENV)

    def delay(self, shard: Shard, config: Config, system: TestSystem) -> None:
        time.sleep(config.gpu_delay / 1000)

    def shard_args(self, shard: Shard, config: Config) -> ArgList:
        raise NotImplementedError()

    def compute_spec(self, config: Config, system: TestSystem) -> StageSpec:
        raise NotImplementedError()
