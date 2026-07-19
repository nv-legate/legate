# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ..data.scalar import Scalar
from ..task.task_info import TaskInfo
from ..type.types import Type
from ..utilities.typedefs import GlobalRedopID, LocalRedopID, LocalTaskID
from ..utilities.unconstructable import Unconstructable

class Library(Unconstructable):
    @property
    def name(self) -> str: ...
    def get_new_task_id(self) -> LocalTaskID: ...
    def get_reduction_op_id(
        self, local_redop_id: LocalRedopID | int
    ) -> GlobalRedopID: ...
    def get_tunable(self, tunable_id: int, dtype: Type) -> Scalar: ...
    def register_task(self, task_info: TaskInfo) -> LocalTaskID: ...
    @property
    def raw_handle(self) -> int: ...
