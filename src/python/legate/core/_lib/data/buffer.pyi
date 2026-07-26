# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Any

from ..geometry.rect import Rect
from ..mapping.mapping import StoreTarget
from ..type.types import Type
from ..utilities.unconstructable import Unconstructable
from .raw_allocation import RawAllocation

class Buffer(Unconstructable):
    @staticmethod
    def allocate(
        dtype: Type, domain: Rect | Sequence[int], target: StoreTarget
    ) -> Buffer: ...
    @property
    def type(self) -> Type: ...
    @property
    def dim(self) -> int: ...
    @property
    def domain(self) -> Rect: ...
    @property
    def target(self) -> StoreTarget: ...
    @property
    def allocation(self) -> RawAllocation: ...
    def reshape(self, new_domain: Rect | Sequence[int]) -> None: ...
    @property
    def __array_interface__(self) -> dict[str, Any]: ...
    @property
    def __cuda_array_interface__(self) -> dict[str, Any]: ...
