# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from libc.stdint cimport int32_t, int64_t, uint32_t, uint64_t
from libcpp cimport bool
from libcpp.string cimport string as std_string
from libcpp.vector cimport vector as std_vector

from ..mapping.mapping cimport StoreTarget
from ..type.type_info cimport _Type
from ..utilities.shared_ptr cimport _SharedPtr
from ..utilities.tuple cimport _tuple
from ..utilities.unconstructable cimport Unconstructable
from .detail.logical_store cimport _LogicalStoreImpl
from .physical_store cimport PhysicalStore, _PhysicalStore
from .shape cimport _Shape
from .slice cimport _Slice


cdef extern from "legate/data/logical_store.h" namespace "legate" nogil:
    cdef cppclass _LogicalStore "legate::LogicalStore":
        _LogicalStore() except+
        _LogicalStore(const _LogicalStore&) except+
        int32_t dim() except+
        bool has_scalar_storage() except+
        bool overlaps(const _LogicalStore&) except+
        _Type type() except+
        _Shape shape() except+
        const _tuple[uint64_t]& extents() except+
        size_t volume() except+
        bool unbound() except+
        bool transformed() except+
        _LogicalStore promote(int32_t, size_t) except+
        _LogicalStore project(int32_t, int64_t) except+
        _LogicalStore slice(int32_t, _Slice) except+
        _LogicalStore transpose(std_vector[int32_t]) except+
        _LogicalStore delinearize(int32_t, std_vector[uint64_t]) except+
        _LogicalStorePartition partition_by_tiling(
            std_vector[uint64_t] tile_shape
        ) except+
        _PhysicalStore get_physical_store() except+
        void detach() except+
        void offload_to(StoreTarget) except+
        std_string to_string() except+
        bool equal_storage(const _LogicalStore&) except+
        const _SharedPtr[_LogicalStoreImpl]& impl() except+

    cdef cppclass _LogicalStorePartition "legate::LogicalStorePartition":
        _LogicalStorePartition() except+
        _LogicalStorePartition(const _LogicalStorePartition&) except+
        _LogicalStore store() except+
        const _tuple[uint64_t]& color_shape() except+
        _LogicalStore get_child_store(const _tuple[uint64_t]&) except+


cdef class LogicalStore(Unconstructable):
    cdef _LogicalStore _handle

    @staticmethod
    cdef LogicalStore from_handle(_LogicalStore)

    cpdef bool overlaps(self, LogicalStore other)

    cpdef LogicalStore promote(self, int32_t extra_dim, size_t dim_size)
    cpdef LogicalStore project(self, int32_t dim, int64_t index)
    cpdef LogicalStore slice(self, int32_t dim, slice sl)
    cpdef LogicalStore transpose(self, object axes)
    cpdef LogicalStore delinearize(self, int32_t dim, tuple shape)
    cpdef void fill(self, object value)
    cpdef LogicalStorePartition partition_by_tiling(self, object shape)
    cpdef PhysicalStore get_physical_store(self)
    cpdef void detach(self)
    cpdef void offload_to(self, StoreTarget target_mem)
    cpdef bool equal_storage(self, LogicalStore other)


cdef class LogicalStorePartition:
    cdef _LogicalStorePartition _handle

    @staticmethod
    cdef LogicalStorePartition from_handle(_LogicalStorePartition)

    cpdef LogicalStore store(self)
