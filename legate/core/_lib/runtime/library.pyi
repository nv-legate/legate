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

from ..data.scalar import Scalar
from ..task.task_info import TaskInfo
from ..type.type_info import Type
from ..utilities.typedefs import (
    GlobalRedopID,
    GlobalTaskID,
    LocalRedopID,
    LocalTaskID,
)
from ..utilities.unconstructable import Unconstructable

class Library(Unconstructable):
    def get_new_task_id(self) -> LocalTaskID: ...
    # This prototype is a lie, technically (in Cython) it's only LocalTaskID,
    # but we allow int as a type-checking convencience to users
    def get_task_id(
        self, local_task_id: int | LocalTaskID
    ) -> GlobalTaskID: ...
    def get_mapper_id(self) -> int: ...
    def get_reduction_op_id(
        self, local_redop_id: LocalRedopID | int
    ) -> GlobalRedopID: ...
    def get_tunable(self, tunable_id: int, dtype: Type) -> Scalar: ...
    def register_task(self, task_info: TaskInfo) -> GlobalTaskID: ...
