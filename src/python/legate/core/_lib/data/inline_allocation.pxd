# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int32_t
from libcpp.vector cimport vector as std_vector

from ..mapping.mapping cimport StoreTarget
from ..type.types cimport _Type
from ..utilities.typedefs cimport _Domain
from .physical_store cimport PhysicalStore


cdef extern from "legate/data/inline_allocation.h" namespace "legate" nogil:
    cdef cppclass _InlineAllocation "legate::InlineAllocation":
        void* ptr
        std_vector[size_t] strides


cdef class InlineAllocation:
    cdef _InlineAllocation _handle
    cdef PhysicalStore _store
    cdef tuple _shape

    @staticmethod
    cdef InlineAllocation create(PhysicalStore store, _InlineAllocation handle)

    cdef dict _get_array_interface(self)
