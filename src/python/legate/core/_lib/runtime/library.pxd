# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int64_t

from ..data.scalar cimport Scalar, _Scalar
from ..task.task_info cimport TaskInfo, _TaskInfo
from ..type.types cimport Type, _Type
from ..utilities.typedefs cimport (
    _GlobalRedopID,
    _GlobalTaskID,
    _LocalRedopID,
    _LocalTaskID,
)
from ..utilities.unconstructable cimport Unconstructable


cdef extern from "legate/runtime/library.h" namespace "legate" nogil:
    cdef cppclass _Library "legate::Library":
        _Library() except+
        _Library(const _Library&) except+
        _LocalTaskID get_new_task_id() except+
        _GlobalTaskID get_task_id(_LocalTaskID) except+
        _GlobalRedopID get_reduction_op_id(_LocalRedopID) except+
        _Scalar get_tunable(int64_t, _Type) except+
        void register_task(_LocalTaskID, const _TaskInfo&) except+


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
