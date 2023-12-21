# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from collections.abc import Callable
from typing import Any

from ...task.type import VariantFunction

class TaskInfo:
    def __dealloc__(self) -> None: ...
    def __repr__(self) -> str: ...
    @classmethod
    def from_variants(
        cls,
        local_task_id: int,
        name: str,
        variants: list[tuple[int, VariantFunction]],
    ) -> TaskInfo: ...
    @property
    def valid(self) -> bool: ...
    @property
    def name(self) -> str: ...
    def has_variant(self, variant_id: int) -> bool: ...
    def add_variant(self, variant_kind: int, fn: VariantFunction) -> None: ...
