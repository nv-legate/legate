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
from libcpp.memory cimport unique_ptr as std_unique_ptr

from ..data.scalar cimport Scalar, _Scalar
from ..task.task_info cimport TaskInfo, _TaskInfo
from ..type.type_info cimport Type, _Type


cdef extern from "core/runtime/library.h" namespace "legate" nogil:
    cdef cppclass _Library "legate::Library":
        _Library()
        _Library(const _Library&)
        int64_t get_new_task_id()
        uint32_t get_task_id(int64_t)
        uint32_t get_mapper_id()
        uint32_t get_reduction_op_id(int64_t)
        _Scalar get_tunable(int64_t, _Type)
        void register_task(int64_t, std_unique_ptr[_TaskInfo])


cdef class Library:
    cdef _Library _handle

    @staticmethod
    cdef Library from_handle(_Library)

    cpdef int64_t get_new_task_id(self)
    cpdef uint32_t get_task_id(self, int64_t local_task_id)
    cpdef uint32_t get_mapper_id(self)
    cpdef uint32_t get_reduction_op_id(self, int64_t local_redop_id)
    cpdef Scalar get_tunable(self, int64_t tunable_id, Type dtype)
    cpdef uint32_t register_task(self, TaskInfo task_info)
