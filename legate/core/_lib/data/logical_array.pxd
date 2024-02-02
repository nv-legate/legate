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

from libc.stdint cimport int32_t, int64_t, uint32_t, uint64_t, uintptr_t
from libcpp cimport bool
from libcpp.vector cimport vector as std_vector

from ..type.type_info cimport _Type
from ..utilities.tuple cimport _tuple
from .logical_store cimport _LogicalStore
from .shape cimport _Shape
from .slice cimport _Slice


cdef extern from "core/data/logical_array.h" namespace "legate" nogil:
    cdef cppclass _LogicalArray "legate::LogicalArray":
        int32_t dim()
        _Type type()
        _Shape shape()
        const _tuple[uint64_t]& extents() except+
        size_t volume() except+
        bool unbound()
        bool nullable()
        bool nested()
        uint32_t num_children()
        _LogicalArray promote(int32_t, size_t) except+
        _LogicalArray project(int32_t, int64_t) except+
        _LogicalArray slice(int32_t, _Slice) except+
        _LogicalArray transpose(std_vector[int32_t]) except+
        _LogicalArray delinearize(int32_t, std_vector[uint64_t]) except+
        _LogicalStore data() except+
        _LogicalStore null_mask() except+
        _LogicalArray child(uint32_t) except+
        _LogicalArray()
        _LogicalArray(const _LogicalStore&)
        _LogicalArray(const _LogicalArray&)


cdef class LogicalArray:
    cdef _LogicalArray _handle

    @staticmethod
    cdef LogicalArray from_handle(_LogicalArray)

    cpdef LogicalArray promote(self, int32_t extra_dim, size_t dim_size)
    cpdef LogicalArray project(self, int32_t dim, int64_t index)
    cpdef LogicalArray slice(self, int32_t dim, slice sl)
    cpdef LogicalArray transpose(self, object axes)
    cpdef LogicalArray delinearize(self, int32_t dim, object shape)
    cpdef LogicalArray child(self, uint32_t index)


cdef _LogicalArray to_cpp_logical_array(object array_or_store)
