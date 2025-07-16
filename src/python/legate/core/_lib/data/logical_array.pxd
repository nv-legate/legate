# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int32_t, int64_t, uint32_t, uint64_t, uintptr_t
from libcpp cimport bool
from libcpp.optional cimport optional as std_optional
from libcpp.vector cimport vector as std_vector

from ..mapping.mapping cimport StoreTarget
from ..type.types cimport _Type
from ..utilities.tuple cimport _tuple
from ..utilities.unconstructable cimport Unconstructable
from .logical_store cimport _LogicalStore
from .physical_array cimport PhysicalArray, _PhysicalArray
from .shape cimport _Shape
from .slice cimport _Slice


cdef extern from "legate/data/logical_array.h" namespace "legate" nogil:
    cdef cppclass _LogicalArray "legate::LogicalArray":
        int32_t dim() except+
        _Type type() except+
        _Shape shape() except+
        _tuple[uint64_t] extents() except+
        size_t volume() except+
        bool unbound() except+
        bool nullable() except+
        bool nested() except+
        uint32_t num_children() except+
        _LogicalArray promote(int32_t, size_t) except+
        _LogicalArray project(int32_t, int64_t) except+
        _LogicalArray slice(int32_t, _Slice) except+
        _LogicalArray transpose(std_vector[int32_t]) except+
        _LogicalArray delinearize(int32_t, std_vector[uint64_t]) except+
        _LogicalStore data() except+
        _LogicalStore null_mask() except+
        _LogicalArray child(uint32_t) except+
        _PhysicalArray get_physical_array(std_optional[StoreTarget]) except+
        void offload_to(StoreTarget) except+
        _LogicalArray()
        _LogicalArray(const _LogicalStore&) except+
        _LogicalArray(const _LogicalArray&) except+


cdef class LogicalArray(Unconstructable):
    cdef _LogicalArray _handle

    @staticmethod
    cdef LogicalArray from_handle(_LogicalArray)

    cpdef LogicalArray promote(self, int32_t extra_dim, size_t dim_size)
    cpdef LogicalArray project(self, int32_t dim, int64_t index)
    cpdef LogicalArray slice(self, int32_t dim, slice sl)
    cpdef LogicalArray transpose(self, object axes)
    cpdef LogicalArray delinearize(self, int32_t dim, object shape)
    cpdef void fill(self, object value)
    cpdef LogicalArray child(self, uint32_t index)
    cpdef PhysicalArray get_physical_array(self, target: object =*)
    cpdef void offload_to(self, StoreTarget target_mem)


cdef _LogicalArray to_cpp_logical_array(object array_or_store)
