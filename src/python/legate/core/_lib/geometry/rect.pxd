# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stddef cimport size_t
from libc.stdint cimport uint32_t
from libcpp cimport bool

from .point cimport _Point

from ..utilities.unconstructable cimport Unconstructable

cdef extern from "legate/geometry/rect.h" namespace "legate" nogil:
    cdef cppclass _Rect "legate::Rect":
        _Rect()
        _Rect(_Point, _Point)
        bool operator==(const _Rect&) except+
        bool operator!=(const _Rect&) except+

        uint32_t dim() except+
        bool empty() except+
        size_t volume() except+
        _Point lo() except+
        _Point hi() except+
        _Point extents() except+
        _Point strides() except+


cdef _Rect _rect_from_iterables(object lo, object hi) except *


cdef class Rect(Unconstructable):
    cdef _Rect _handle

    @staticmethod
    cdef Rect from_handle(_Rect handle)
