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

from typing import Any

from ..mapping.mapping import StoreTarget
from ..type.type_info import Type
from ..utilities.typedefs import Domain
from .inline_allocation import InlineAllocation

class PhysicalStore:
    @property
    def ndim(self) -> int: ...
    @property
    def type(self) -> Type: ...
    @property
    def domain(self) -> Domain: ...
    @property
    def target(self) -> StoreTarget: ...
    def get_inline_allocation(self) -> InlineAllocation: ...
