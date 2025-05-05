# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool
from libc.stdint cimport int32_t

from .inline_allocation cimport _InlineAllocation, InlineAllocation

from ..type.types cimport _Type
from ..utilities.typedefs cimport _Domain
from ..utilities.unconstructable cimport Unconstructable

cdef extern from "legate/data/buffer.h" namespace "legate" nogil:
    cdef cppclass _TaskLocalBuffer "legate::TaskLocalBuffer":
        _TaskLocalBuffer() except+

        _Type type() except+
        int32_t dim() except+
        const _Domain& domain() except+

        _InlineAllocation get_inline_allocation() except+


cdef class TaskLocalBuffer(Unconstructable):
    cdef _TaskLocalBuffer _handle
    cdef InlineAllocation _alloc
    cdef object _owner

    @staticmethod
    cdef TaskLocalBuffer from_handle(
        const _TaskLocalBuffer &handle, object owner=*
    )

    cdef InlineAllocation _init_inline_allocation(self)
    cdef InlineAllocation _get_inline_allocation(self)
