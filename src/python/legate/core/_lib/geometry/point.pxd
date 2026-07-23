# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stddef cimport size_t
from libc.stdint cimport int32_t, int64_t, uint32_t
from libcpp cimport bool
from libcpp.vector cimport vector as std_vector


cdef extern from "legate/geometry/point.h" namespace "legate" nogil:
    cdef cppclass _Point "legate::Point":
        _Point()
        uint32_t dim() except+
        size_t size() except+
        bool empty() except+
        int64_t volume() except+

        bool operator==(const _Point&) except+
        bool operator!=(const _Point&) except+

        int64_t& operator[](int32_t) except+


cdef extern from * nogil:
    """
    #include <legate/geometry/point.h>
    #include <legate/utilities/span.h>

    #include <cstdint>
    #include <vector>

    namespace {

    legate::Point legate_point_from_coords(
      const std::vector<std::int64_t>& coords)
    {
      return legate::Point{
        legate::Span<const std::int64_t>{coords.data(), coords.size()}};
    }

    }  // namespace
    """
    _Point _point_from_coords "legate_point_from_coords" (
        const std_vector[int64_t]&
    ) except+


cdef _Point _point_from_iterable(object obj) except *


cdef class Point:
    cdef _Point _handle

    @staticmethod
    cdef Point from_handle(_Point handle)
