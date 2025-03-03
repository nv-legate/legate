# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from ..mapping.mapping import StoreTarget
from ..type.types import Type
from ..utilities.typedefs import Domain
from ..utilities.unconstructable import Unconstructable
from .inline_allocation import InlineAllocation

class PhysicalStore(Unconstructable):
    @property
    def ndim(self) -> int: ...
    @property
    def type(self) -> Type: ...
    @property
    def domain(self) -> Domain: ...
    @property
    def target(self) -> StoreTarget: ...
    def get_inline_allocation(self) -> InlineAllocation: ...
    @property
    def __array_interface__(self) -> dict[str, Any]: ...
    @property
    def __cuda_array_interface__(self) -> dict[str, Any]: ...
