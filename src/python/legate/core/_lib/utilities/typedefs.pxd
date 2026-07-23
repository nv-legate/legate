# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from libc.stdint cimport int64_t

from ..task.task_context cimport _TaskContext


cdef extern from "legate/utilities/typedefs.h" namespace "legate" nogil:
    cdef enum class _LocalTaskID "legate::LocalTaskID" (int64_t):
        pass

    ctypedef int _Legion_ReductionOpID "Legion::ReductionOpID"

    cdef enum class _LocalRedopID "legate::LocalRedopID" (int64_t):
        pass

    cdef enum class _GlobalRedopID "legate::GlobalRedopID" (
        _Legion_ReductionOpID
    ):
        pass

    cdef cppclass _Processor "legate::Processor":
        pass

    cpdef enum class VariantCode:
        CPU
        GPU
        OMP

# note missing nogil!
cdef extern from "legate/utilities/typedefs.h" namespace "legate":
    ctypedef void (*VariantImpl)(_TaskContext) except+
