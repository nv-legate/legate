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

from libc.stdint cimport int64_t, uint32_t

from ..data.scalar cimport Scalar
from ..type.type_info cimport Type


cdef class Library:
    @staticmethod
    cdef Library from_handle(_Library handle):
        cdef Library result = Library.__new__(Library)
        result._handle = handle
        return result

    def get_task_id(self, int64_t local_task_id) -> uint32_t:
        return self._handle.get_task_id(local_task_id)

    def get_mapper_id(self) -> uint32_t:
        return self._handle.get_mapper_id()

    def get_reduction_op_id(self, int64_t local_redop_id) -> uint32_t:
        return self._handle.get_reduction_op_id(local_redop_id)

    def get_tunable(self, int64_t tunable_id, Type dtype) -> Scalar:
        return Scalar.from_handle(
            self._handle.get_tunable(tunable_id, dtype._handle)
        )
