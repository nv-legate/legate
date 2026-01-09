# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum, unique
from typing import cast

from ..utilities.unconstructable import Unconstructable

@unique
class TaskTarget(IntEnum):
    GPU = cast(int, ...)
    OMP = cast(int, ...)
    CPU = cast(int, ...)

@unique
class StoreTarget(IntEnum):
    """Specify what memory to offload objects to."""  # noqa: PYI021

    SYSMEM = cast(int, ...)
    FBMEM = cast(int, ...)
    ZCMEM = cast(int, ...)
    SOCKETMEM = cast(int, ...)

class DimOrdering(Unconstructable):
    @unique
    class Kind(IntEnum):
        C = cast(int, ...)
        FORTRAN = cast(int, ...)
        CUSTOM = cast(int, ...)

    @staticmethod
    def c_order() -> DimOrdering: ...
    @staticmethod
    def fortran_order() -> DimOrdering: ...
    @staticmethod
    def custom_order(dims: list[int]) -> DimOrdering: ...
    @property
    def kind(self) -> Kind: ...
