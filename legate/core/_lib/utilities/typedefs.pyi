# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from enum import Enum
from typing import NewType

LocalTaskID = NewType("LocalTaskID", int)
GlobalTaskID = NewType("GlobalTaskID", int)

LocalRedopID = NewType("LocalRedopID", int)
GlobalRedopID = NewType("GlobalRedopID", int)

class VariantCode(Enum):
    NONE: int
    CPU: int
    GPU: int
    OMP: int

class DomainPoint:
    def __init__(self) -> None: ...
    @property
    def dim(self) -> int: ...
    def __getitem__(self, idx: int) -> int: ...
    def __setitem__(self, idx: int, coord: int) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __str__(self) -> str: ...

class Domain:
    def __init__(self) -> None: ...
    @property
    def dim(self) -> int: ...
    @property
    def lo(self) -> DomainPoint: ...
    @property
    def hi(self) -> DomainPoint: ...
    def __eq__(self, other: object) -> bool: ...
    def __str__(self) -> str: ...
