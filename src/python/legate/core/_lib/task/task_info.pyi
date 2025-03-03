# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ..._ext.task.type import VariantFunction
from ..utilities.typedefs import LocalTaskID, VariantCode
from ..utilities.unconstructable import Unconstructable

class TaskInfo(Unconstructable):
    @classmethod
    def from_variants(
        cls,
        local_task_id: LocalTaskID,
        name: str,
        variants: list[tuple[VariantCode, VariantFunction]],
    ) -> TaskInfo: ...
    @property
    def name(self) -> str: ...
    def has_variant(self, variant_id: VariantCode) -> bool: ...
    def add_variant(
        self, variant_kind: VariantCode, fn: VariantFunction
    ) -> None: ...
