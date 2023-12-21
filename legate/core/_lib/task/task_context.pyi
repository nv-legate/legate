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

from ..data.physical_array import PhysicalArray
from ..data.scalar import Scalar

class TaskContext:
    def get_task_id(self) -> int: ...
    def get_variant_kind(self) -> int: ...
    @property
    def inputs(self) -> tuple[PhysicalArray, ...]: ...
    @property
    def outputs(self) -> tuple[PhysicalArray, ...]: ...
    @property
    def reductions(self) -> tuple[PhysicalArray, ...]: ...
    @property
    def scalars(self) -> tuple[Scalar, ...]: ...
    def set_exception(self, excn: Exception) -> None: ...
