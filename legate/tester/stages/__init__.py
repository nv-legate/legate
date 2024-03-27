# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Provide TestStage subclasses for running configured test files using
specific features.

"""
from __future__ import annotations

import sys

from .. import FeatureType
from .test_stage import TestStage
from .util import log_proc

if sys.platform == "darwin":
    from ._osx import CPU, Eager, GPU, OMP
elif sys.platform.startswith("linux"):
    from ._linux import CPU, Eager, GPU, OMP
else:
    raise RuntimeError(f"unsupported platform: {sys.platform}")

#: All the available test stages that can be selected
STAGES: dict[FeatureType, type[TestStage]] = {
    "cpus": CPU,
    "cuda": GPU,
    "openmp": OMP,
    "eager": Eager,
}
