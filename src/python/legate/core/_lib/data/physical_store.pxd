# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool
from libc.stdint cimport int32_t

from ..geometry.point cimport _Point
from ..geometry.rect cimport _Rect
from ..mapping.mapping cimport StoreTarget
from ..type.types cimport _Type
from ..utilities.unconstructable cimport Unconstructable
from .raw_allocation cimport _RawAllocation, RawAllocation
from .buffer cimport _Buffer, Buffer
from .logical_store cimport _LogicalStore, LogicalStore

cdef extern from "legate/data/physical_store.h" namespace "legate" nogil:
    cdef cppclass _PhysicalStore "legate::PhysicalStore":
        int32_t dim() except+
        _Type type() except+
        _Rect domain() except+
        _Buffer get_buffer() except+
        void* base() except+
        const _Point& strides() except+
        StoreTarget target() except+
        _Buffer create_output_buffer(
            const _Point& extents, bool bind_buffer
        ) except+
        void bind_data(
            const _Buffer& buffer, const _Point& extents
        ) except+


cdef class PhysicalStore(Unconstructable):
    cdef _PhysicalStore _handle

    @staticmethod
    cdef PhysicalStore from_handle(_PhysicalStore)

    cpdef Buffer create_output_buffer(
        self, object shape, bool bind = *
    )
    cpdef void bind_data(self, Buffer buffer, object extent = *)
    cpdef Buffer get_buffer(self)
    cdef dict _array_interface(self)

    cpdef tuple[int32_t, int32_t] __dlpack_device__(self)
