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

from libc.stdint cimport int32_t, int64_t, uint32_t, uint64_t
from libcpp cimport bool
from libcpp.string cimport string as std_string
from libcpp.vector cimport vector as std_vector

from ..type.type_info cimport _Type
from ..utilities.shared_ptr cimport _SharedPtr
from .detail.logical_store cimport _LogicalStoreImpl
from .physical_store cimport _PhysicalStore
from .shape cimport _Shape
from .slice cimport _Slice


cdef extern from "core/data/logical_store.h" namespace "legate" nogil:
    cdef cppclass _LogicalStore "legate::LogicalStore":
        _LogicalStore()
        _LogicalStore(const _LogicalStore&)
        int32_t dim()
        bool has_scalar_storage()
        bool overlaps(const _LogicalStore&)
        _Type type()
        const _Shape& extents() except+
        size_t volume() except+
        bool unbound()
        bool transformed()
        _LogicalStore promote(int32_t, size_t) except+
        _LogicalStore project(int32_t, int64_t) except+
        _LogicalStore slice(int32_t, _Slice) except+
        _LogicalStore transpose(std_vector[int32_t]) except+
        _LogicalStore delinearize(int32_t, std_vector[uint64_t]) except+
        _LogicalStorePartition partition_by_tiling(
            std_vector[size_t] tile_shape
        )
        _PhysicalStore get_physical_store()
        void detach()
        std_string to_string()
        const _SharedPtr[_LogicalStoreImpl]& impl() const

    cdef cppclass _LogicalStorePartition "legate::LogicalStorePartition":
        _LogicalStorePartition()
        _LogicalStorePartition(const _LogicalStorePartition&)
        _LogicalStore store()
        const _Shape& color_shape() except+
        _LogicalStore get_child_store(const _Shape&)


cdef class LogicalStore:
    cdef _LogicalStore _handle

    @staticmethod
    cdef LogicalStore from_handle(_LogicalStore)


cdef class LogicalStorePartition:
    cdef _LogicalStorePartition _handle

    @staticmethod
    cdef LogicalStorePartition from_handle(_LogicalStorePartition)
