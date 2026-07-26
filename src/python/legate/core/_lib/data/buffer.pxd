# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool
from libcpp.pair cimport pair as std_pair
from libc.stdint cimport int32_t

from ..geometry.point cimport _Point
from ..geometry.rect cimport _Rect
from ..mapping.mapping cimport StoreTarget
from ..type.types cimport _Type
from ..utilities.unconstructable cimport Unconstructable
from .raw_allocation cimport _RawAllocation

cdef extern from "legate/data/buffer.h" namespace "legate" nogil:
    cdef cppclass _Buffer "legate::Buffer":
        _Buffer() except+

        _Type type() except+
        int32_t dim() except+
        const _Rect& domain() except+
        StoreTarget target() except+
        _RawAllocation allocation() except+
        void reshape(_Rect) except+
        std_pair[void*, _Point] base_and_element_strides() except+

        @staticmethod
        _Buffer allocate(const _Type&, _Rect, StoreTarget) except+


cdef class Buffer(Unconstructable):
    cdef _Buffer _handle
    cdef object _owner

    @staticmethod
    cdef Buffer from_handle(
        const _Buffer &handle, object owner=*
    )

    cdef dict _array_interface(self)
