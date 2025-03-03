# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from libc.stddef cimport size_t
from libc.stdint cimport int32_t, int64_t, uint16_t
from libcpp cimport bool

from ..task.task_context cimport _TaskContext


cdef extern from "legion.h" nogil:
    cdef cppclass __half:
        __half()
        __half(float)
        uint16_t raw() const

    float __convert_halfint_to_float(uint16_t)

cdef extern from "legate/utilities/typedefs.h" namespace "legate" nogil:
    ctypedef unsigned int _Legion_TaskID "Legion::TaskID"

    cdef enum class _LocalTaskID "legate::LocalTaskID" (int64_t):
        pass

    cdef enum class _GlobalTaskID "legate::GlobalTaskID" (_Legion_TaskID):
        pass

    ctypedef int _Legion_ReductionOpID "Legion::ReductionOpID"

    cdef enum class _LocalRedopID "legate::LocalRedopID" (int64_t):
        pass

    cdef enum class _GlobalRedopID "legate::GlobalRedopID" (
        _Legion_ReductionOpID
    ):
        pass

    cdef cppclass _DomainPoint "legate::DomainPoint":
        _DomainPoint() except+
        int32_t get_dim() except+
        int64_t& operator[](int32_t) except+
        bool operator==(const _DomainPoint&) except+

    cdef cppclass _Domain "legate::Domain":
        _Domain() except+
        _Domain(const _DomainPoint&, const _DomainPoint) except+
        int32_t get_dim() except+
        _DomainPoint lo() except+
        _DomainPoint hi() except+
        bool operator==(const _Domain&) except+

    cdef cppclass _Processor "legate::Processor":
        pass

    cpdef enum class VariantCode:
        CPU
        GPU
        OMP

# note missing nogil!
cdef extern from "legate/utilities/typedefs.h" namespace "legate":
    ctypedef void (*VariantImpl)(_TaskContext) except+
    ctypedef void (*TaskFuncPtr "legate::Processor::TaskFuncPtr")(
        const void *, size_t, const void *, size_t, _Processor
    ) except+

cdef class DomainPoint:
    cdef _DomainPoint _handle

    @staticmethod
    cdef DomainPoint from_handle(_DomainPoint)

cdef class Domain:
    cdef _Domain _handle

    @staticmethod
    cdef Domain from_handle(_Domain)


cdef extern from * nogil:
    """
    namespace {

    [[nodiscard]] float half_to_float(__half h)
    {
      return static_cast<float>(h);
    }

    } // namespace
    """
    float half_to_float(__half)
