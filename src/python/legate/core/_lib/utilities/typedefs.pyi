# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum
from typing import NewType, cast

LocalTaskID = NewType("LocalTaskID", int)

LocalRedopID = NewType("LocalRedopID", int)
GlobalRedopID = NewType("GlobalRedopID", int)

class VariantCode(IntEnum):
    CPU = cast(int, ...)
    GPU = cast(int, ...)
    OMP = cast(int, ...)
