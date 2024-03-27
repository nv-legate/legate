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

# This file exists because we want to keep "from legate.core.task import foo"
# because the extra namespace helps to disambiguate things. I tried doing:
#
# from ._ext import task
#
# in legate/core/__init__.py but Python doesn't like that, and in fact, won't
# consider attributes as modules during module lookup:
#
# ModuleNotFoundError: No module named 'legate.core.task'
#
# So the only solution is to keep a dummy "module" here, whose only job is to
# mirror the real module over in _ext/task.
from .._ext.task import (
    task,
    PyTask,
    VariantInvoker,
    InputStore,
    OutputStore,
    ReductionStore,
    InputArray,
    OutputArray,
)

__all__ = (
    "task",
    "PyTask",
    "VariantInvoker",
    "InputStore",
    "OutputStore",
    "ReductionStore",
    "InputArray",
    "OutputArray",
)

# Not in __all__, this is intentional! This module is only "exposed" for
# testing purposes.
from .._ext.task import util as _util
