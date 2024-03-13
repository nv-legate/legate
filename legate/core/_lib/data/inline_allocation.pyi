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

class InlineAllocation:
    @property
    def ptr(self) -> int: ...
    @property
    def strides(self) -> tuple[int, ...]: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def __array_interface__(self) -> dict[str, Any]: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
