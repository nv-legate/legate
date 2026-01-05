# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int64_t
from libcpp.optional cimport optional as std_optional


cdef extern from "legate/data/slice.h" namespace "legate" nogil:
    cdef std_optional[int64_t] OPEN "legate::Slice::OPEN"

    cdef cppclass _Slice "legate::Slice":
        _Slice() except+
        _Slice(std_optional[int64_t], std_optional[int64_t]) except+


cdef _Slice from_python_slice(slice sl)
