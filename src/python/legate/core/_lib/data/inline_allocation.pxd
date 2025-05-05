# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp.vector cimport vector as std_vector

from ..mapping.mapping cimport StoreTarget
from ..type.types cimport Type
from ..utilities.typedefs cimport _Domain
from ..utilities.unconstructable cimport Unconstructable
from .physical_store cimport PhysicalStore


cdef extern from "legate/data/inline_allocation.h" namespace "legate" nogil:
    cdef cppclass _InlineAllocation "legate::InlineAllocation":
        void* ptr
        std_vector[size_t] strides
        StoreTarget target

cdef class InlineAllocation(Unconstructable):
    cdef _InlineAllocation _handle
    cdef object _owner
    cdef dict _array_interface

    @staticmethod
    cdef InlineAllocation create(
        _InlineAllocation handle,
        Type ty,
        tuple shape,
        tuple strides,
        object owner,
    )

    @staticmethod
    cdef tuple _compute_shape(const _Domain& domain)
