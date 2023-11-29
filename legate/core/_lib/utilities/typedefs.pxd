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
from libc.stddef cimport size_t
from libc.stdint cimport int32_t, int64_t

from ..task.task_context cimport _TaskContext


cdef extern from "core/utilities/typedefs.h" namespace "legate" nogil:
    cdef cppclass _DomainPoint "legate::DomainPoint":
        _DomainPoint()
        int32_t get_dim()
        int64_t& operator[](int32_t)

    cdef cppclass _Domain "legate::Domain":
        _Domain()
        _Domain(const _DomainPoint&, const _DomainPoint)
        int32_t get_dim()
        _DomainPoint lo()
        _DomainPoint hi()

    cdef cppclass _Processor "legate::Processor":
        pass

# note missing nogil!
cdef extern from "core/utilities/typedefs.h" namespace "legate":
    ctypedef void (*VariantImpl)(_TaskContext) except +
    ctypedef void (*RealmCallbackFn)(
        const void *, size_t, const void *, size_t, _Processor
    ) except +

cdef class DomainPoint:
    cdef _DomainPoint _handle

    @staticmethod
    cdef DomainPoint from_handle(_DomainPoint)

cdef class Domain:
    cdef _Domain _handle

    @staticmethod
    cdef Domain from_handle(_Domain)
