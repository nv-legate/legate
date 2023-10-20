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

from ..data.scalar import Scalar
from ..type.type_info import Type

class Library:
    def get_task_id(self, local_task_id: int) -> int: ...
    def get_mapper_id(self) -> int: ...
    def get_reduction_op_id(self, local_redop_id: int) -> int: ...
    def get_tunable(self, tunable_id: int, dtype: Type) -> Scalar: ...
