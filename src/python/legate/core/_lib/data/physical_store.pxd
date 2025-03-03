# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int32_t
from libcpp.vector cimport vector as std_vector

from ..mapping.mapping cimport StoreTarget
from ..type.types cimport _Type
from ..utilities.typedefs cimport _Domain
from ..utilities.unconstructable cimport Unconstructable
from .inline_allocation cimport InlineAllocation, _InlineAllocation


cdef extern from "legate/data/physical_store.h" namespace "legate" nogil:
    cdef cppclass _PhysicalStore "legate::PhysicalStore":
        int32_t dim() except+
        _Type type() except+
        _Domain domain() except+
        _InlineAllocation get_inline_allocation() except+
        StoreTarget target() except+


cdef class PhysicalStore(Unconstructable):
    cdef _PhysicalStore _handle

    @staticmethod
    cdef PhysicalStore from_handle(_PhysicalStore)

    cpdef InlineAllocation get_inline_allocation(self)
