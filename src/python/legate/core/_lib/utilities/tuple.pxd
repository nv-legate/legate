# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stddef cimport size_t
from libcpp.vector cimport vector as std_vector


cdef extern from "legate/utilities/tuple.h" namespace "legate" nogil:
    cdef cppclass _tuple "legate::tuple" [T]:
        void append_inplace(const T& value) except+
        const std_vector[T]& data() except+
        void reserve(size_t) except+
