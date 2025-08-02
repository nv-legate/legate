# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp.vector cimport vector as std_vector


cdef extern from "legate/utilities/tuple.h" namespace "legate" nogil:
    cdef cppclass _Span "legate::Span" [T]:
        _Span() except+


cdef extern from * nogil:
    """
    namespace {

    template <typename T>
    std::vector<T> span_to_vector(legate::Span<const T> span)
    {
      return {span.begin(), span.end()};
    }

    } // namespace
    """
    std_vector[T] _span_to_vector "span_to_vector" [T](
        _Span[const T] span
    ) except+
