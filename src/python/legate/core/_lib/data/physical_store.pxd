# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool
from libc.stdint cimport int32_t

from ..mapping.mapping cimport StoreTarget
from ..type.types cimport _Type
from ..utilities.typedefs cimport _Domain, _DomainPoint
from ..utilities.unconstructable cimport Unconstructable
from .inline_allocation cimport InlineAllocation, _InlineAllocation
from .buffer cimport _TaskLocalBuffer, TaskLocalBuffer

cdef extern from "legate/data/physical_store.h" namespace "legate" nogil:
    cdef cppclass _PhysicalStore "legate::PhysicalStore":
        int32_t dim() except+
        _Type type() except+
        _Domain domain() except+
        _InlineAllocation get_inline_allocation() except+
        StoreTarget target() except+
        _TaskLocalBuffer create_output_buffer(
            const _DomainPoint& extents, bool bind_buffer
        ) except+
        void bind_data(
            const _TaskLocalBuffer& buffer, const _DomainPoint& extents
        ) except+


cdef class PhysicalStore(Unconstructable):
    cdef _PhysicalStore _handle

    @staticmethod
    cdef PhysicalStore from_handle(_PhysicalStore)

    cpdef TaskLocalBuffer create_output_buffer(
        self, object shape, bool bind = *
    )
    cpdef void bind_data(self, TaskLocalBuffer buffer, object extent = *)
    cpdef InlineAllocation get_inline_allocation(self)

    cpdef tuple[int32_t, int32_t] __dlpack_device__(self)
