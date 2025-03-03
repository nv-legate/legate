# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum, unique
from typing import cast

@unique
class ExceptionMode(IntEnum):
    IMMEDIATE = cast(int, ...)
    DEFERRED = cast(int, ...)
    IGNORED = cast(int, ...)
