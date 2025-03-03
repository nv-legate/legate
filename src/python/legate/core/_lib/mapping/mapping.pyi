# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum, unique
from typing import cast

@unique
class TaskTarget(IntEnum):
    GPU = cast(int, ...)
    OMP = cast(int, ...)
    CPU = cast(int, ...)

@unique
class StoreTarget(IntEnum):
    SYSMEM = cast(int, ...)
    FBMEM = cast(int, ...)
    ZCMEM = cast(int, ...)
    SOCKETMEM = cast(int, ...)
