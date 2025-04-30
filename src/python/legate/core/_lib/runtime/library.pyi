# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ..data.scalar import Scalar
from ..task.task_info import TaskInfo
from ..type.types import Type
from ..utilities.typedefs import (
    GlobalRedopID,
    GlobalTaskID,
    LocalRedopID,
    LocalTaskID,
)
from ..utilities.unconstructable import Unconstructable

class Library(Unconstructable):
    @property
    def name(self) -> str: ...
    def get_new_task_id(self) -> LocalTaskID: ...
    # This prototype is a lie, technically (in Cython) it's only LocalTaskID,
    # but we allow int as a type-checking convenience to users
    def get_task_id(
        self, local_task_id: LocalTaskID | int
    ) -> GlobalTaskID: ...
    def get_reduction_op_id(
        self, local_redop_id: LocalRedopID | int
    ) -> GlobalRedopID: ...
    def get_tunable(self, tunable_id: int, dtype: Type) -> Scalar: ...
    def register_task(self, task_info: TaskInfo) -> GlobalTaskID: ...
    @property
    def raw_handle(self) -> int: ...
