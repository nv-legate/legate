# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint32_t, uintptr_t
from libcpp cimport bool
from libcpp.optional cimport optional as std_optional

from ..geometry.point cimport _Point
from ..geometry.rect cimport _Rect
from ..mapping.mapping cimport StoreTarget, DimOrderingKind


cdef extern from "legate/data/raw_allocation.h" namespace "legate" nogil:
    cdef cppclass _Deleter "legate::RawAllocation::Deleter":
        pass

    cdef cppclass _RawAllocation "legate::RawAllocation":
        _RawAllocation() except+
        bool read_only() except+
        StoreTarget target() except+
        void* ptr() except+
        size_t size() except+

        @staticmethod
        _RawAllocation create_sysmem(
            void*, size_t, bool, std_optional[_Deleter]
        ) except+

        @staticmethod
        _RawAllocation create_fbmem(
            uint32_t, void*, size_t, bool, std_optional[_Deleter]
        ) except+

        @staticmethod
        _RawAllocation create_zcmem(
            void*, size_t, bool, std_optional[_Deleter]
        ) except+


# Builds a numpy/cupy ``__array_interface__`` dict from a store/buffer's affine
# layout: the origin base pointer, per-dimension element strides, and domain.
# The data pointer is resolved to the lower-bound element and the strides are
# reported in bytes.
cdef dict compute_array_interface(
    str typestr,
    uint32_t field_size,
    const _Rect& domain,
    uintptr_t base,
    const _Point& elem_strides,
)


cdef _RawAllocation create_from_buffer(
    object obj, size_t size, bool read_only, DimOrderingKind order_type
)

cdef _RawAllocation create_from_pointer(
    size_t ptr, size_t size, bool read_only, StoreTarget target,
    uint32_t device_id=*,
    object source=*,
)


cdef class RawAllocation:
    cdef _RawAllocation _handle

    @staticmethod
    cdef RawAllocation from_handle(_RawAllocation handle)
