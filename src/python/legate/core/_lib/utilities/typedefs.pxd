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
        _DomainPoint(int64_t) except+
        int32_t get_dim() except+
        int64_t& operator[](int32_t) except+
        bool operator==(const _DomainPoint&) except+

    cdef cppclass _Domain "legate::Domain":
        _Domain() except+
        _Domain(const _DomainPoint&, const _DomainPoint&) except+
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


# Need to _t these because in the .pyx we also need to define
#
# DomainPoint = tuple[int, ...]
# Domain = tuple[DomainPoint, DomainPoint]
#
# So that people can do "from legate.core import DomainPoint" etc. If we call
# these typdefs "DomainPoint" and "Domain", then Cython complains that the
# above "cannot assign to non-lvalue DomainPoint".
ctypedef tuple[int, ...] DomainPoint_t
ctypedef object Domain_t  # namedtuple

cdef _DomainPoint domain_point_from_iterable(object iterable)
cdef DomainPoint_t domain_point_to_py(const _DomainPoint& point)
cdef _Domain domain_from_iterables(object low, object high)
cdef Domain_t domain_to_py(const _Domain& domain)

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
