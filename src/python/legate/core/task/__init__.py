# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

# This file exists because we want to keep "from legate.task import foo"
# because the extra namespace helps to disambiguate things. I tried doing:
#
# from ._ext import task
#
# in legate/__init__.py but Python doesn't like that, and in fact, won't
# consider attributes as modules during module lookup:
#
# ModuleNotFoundError: No module named 'legate.task'
#
# So the only solution is to keep a dummy "module" here, whose only job is to
# mirror the real module over in _ext/task.
from .._ext.task import (
    ADD,
    AND,
    MAX,
    MIN,
    MUL,
    OR,
    XOR,
    InputArray,
    InputStore,
    OutputArray,
    OutputStore,
    PyTask,
    ReductionArray,
    ReductionStore,
    VariantInvoker,
    task,
)

__all__ = (
    "ADD",
    "AND",
    "MAX",
    "MIN",
    "MUL",
    "OR",
    "XOR",
    "InputArray",
    "InputStore",
    "OutputArray",
    "OutputStore",
    "PyTask",
    "ReductionArray",
    "ReductionStore",
    "VariantInvoker",
    "task",
)

# Not in __all__, this is intentional! This module is only "exposed" for
# testing purposes.
from .._ext.task import util as _util
