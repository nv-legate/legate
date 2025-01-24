# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
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

from . import cmake, package, util
from .cmake import (
    CMAKE_VARIABLE,
    CMakeBool,
    CMakeExecutable,
    CMakeInt,
    CMakeList,
    CMakePath,
    CMaker,
    CMakeString,
)
from .main import basic_configure
from .manager import ConfigurationManager
from .package import MainPackage, Package
from .util.argument_parser import ArgSpec, ConfigArgument

__all__ = (
    "CMAKE_VARIABLE",
    "ArgSpec",
    "CMakeBool",
    "CMakeExecutable",
    "CMakeInt",
    "CMakeList",
    "CMakePath",
    "CMakeString",
    "CMaker",
    "ConfigArgument",
    "ConfigurationManager",
    "MainPackage",
    "Package",
    "basic_configure",
    "cmake",
    "package",
    "util",
)
