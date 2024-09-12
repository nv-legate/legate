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

from libc.stdint cimport int64_t
from libcpp.memory cimport unique_ptr as std_unique_ptr

from ..data.scalar cimport Scalar, _Scalar
from ..task.task_info cimport TaskInfo, _TaskInfo
from ..type.type_info cimport Type, _Type
from ..utilities.typedefs cimport (
    _GlobalRedopID,
    _GlobalTaskID,
    _LocalRedopID,
    _LocalTaskID,
)
from ..utilities.unconstructable cimport Unconstructable


cdef extern from "legate/runtime/library.h" namespace "legate" nogil:
    cdef cppclass _Library "legate::Library":
        _Library()
        _Library(const _Library&)
        _LocalTaskID get_new_task_id()
        _GlobalTaskID get_task_id(_LocalTaskID)
        _GlobalRedopID get_reduction_op_id(_LocalRedopID)
        _Scalar get_tunable(int64_t, _Type)
        void register_task(_LocalTaskID, std_unique_ptr[_TaskInfo])


cdef class Library(Unconstructable):
    cdef _Library _handle

    @staticmethod
    cdef Library from_handle(_Library)

    cpdef _LocalTaskID get_new_task_id(self)
    cpdef _GlobalTaskID get_task_id(self, _LocalTaskID local_task_id)
    cpdef _GlobalRedopID get_reduction_op_id(
        self, _LocalRedopID local_redop_id
    )
    cpdef Scalar get_tunable(self, int64_t tunable_id, Type dtype)
    cpdef _GlobalTaskID register_task(self, TaskInfo task_info)
