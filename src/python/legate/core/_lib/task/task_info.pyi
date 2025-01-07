# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from ..._ext.task.type import VariantFunction
from ..utilities.typedefs import LocalTaskID, VariantCode
from ..utilities.unconstructable import Unconstructable

class TaskInfo(Unconstructable):
    def __dealloc__(self) -> None: ...
    @classmethod
    def from_variants(
        cls,
        local_task_id: LocalTaskID,
        name: str,
        variants: list[tuple[VariantCode, VariantFunction]],
    ) -> TaskInfo: ...
    @property
    def valid(self) -> bool: ...
    @property
    def name(self) -> str: ...
    def has_variant(self, variant_id: VariantCode) -> bool: ...
    def add_variant(
        self, variant_kind: VariantCode, fn: VariantFunction
    ) -> None: ...
