# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum
from typing import NewType, cast

LocalTaskID = NewType("LocalTaskID", int)
GlobalTaskID = NewType("GlobalTaskID", int)

LocalRedopID = NewType("LocalRedopID", int)
GlobalRedopID = NewType("GlobalRedopID", int)

class VariantCode(IntEnum):
    CPU = cast(int, ...)
    GPU = cast(int, ...)
    OMP = cast(int, ...)

class DomainPoint:
    def __init__(self) -> None: ...
    @property
    def dim(self) -> int: ...
    def __getitem__(self, idx: int) -> int: ...
    def __setitem__(self, idx: int, coord: int) -> None: ...
    def __eq__(self, other: object) -> bool: ...

class Domain:
    def __init__(self) -> None: ...
    @property
    def dim(self) -> int: ...
    @property
    def lo(self) -> DomainPoint: ...
    @property
    def hi(self) -> DomainPoint: ...
    def __eq__(self, other: object) -> bool: ...
