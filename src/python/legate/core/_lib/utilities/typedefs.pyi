# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum
from typing import NamedTuple, NewType, TypeAlias, cast

LocalTaskID = NewType("LocalTaskID", int)
GlobalTaskID = NewType("GlobalTaskID", int)

LocalRedopID = NewType("LocalRedopID", int)
GlobalRedopID = NewType("GlobalRedopID", int)

class VariantCode(IntEnum):
    CPU = cast(int, ...)
    GPU = cast(int, ...)
    OMP = cast(int, ...)

DomainPoint: TypeAlias = tuple[int, ...]

class Domain(NamedTuple):
    lo: DomainPoint = (0,)
    hi: DomainPoint = (0,)
