# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint32_t
from libcpp cimport bool
from libcpp.optional cimport optional as std_optional

from ..mapping.mapping cimport TaskTarget, StoreTarget, DimOrderingKind


cdef extern from "legate/data/external_allocation.h" namespace "legate" nogil:
    cdef cppclass _Deleter "legate::ExternalAllocation::Deleter":
        pass

    cdef cppclass _ExternalAllocation "legate::ExternalAllocation":
        _ExternalAllocation() except+
        bool read_only() except+
        TaskTarget target() except+
        void* ptr() except+
        size_t size() except+

        @staticmethod
        _ExternalAllocation create_sysmem(
            void*, size_t, bool, std_optional[_Deleter]
        ) except+

        @staticmethod
        _ExternalAllocation create_fbmem(
            uint32_t, void*, size_t, bool, std_optional[_Deleter]
        ) except+

        @staticmethod
        _ExternalAllocation create_zcmem(
            void*, size_t, bool, std_optional[_Deleter]
        ) except+


cdef _ExternalAllocation create_from_buffer(
    object obj, size_t size, bool read_only, DimOrderingKind order_type
)

cdef _ExternalAllocation create_from_pointer(
    size_t ptr, size_t size, bool read_only, StoreTarget target,
    uint32_t device_id=*,
    object source=*,
)


cdef class ExternalAllocation:
    cdef _ExternalAllocation _handle

    @staticmethod
    cdef ExternalAllocation from_handle(_ExternalAllocation handle)
