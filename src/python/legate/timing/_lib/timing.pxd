# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int64_t

from ...core._lib.utilities.unconstructable cimport Unconstructable


cdef extern from "legate/timing/timing.h" namespace "legate::timing" nogil:
    cdef cppclass _Time "legate::timing::Time":
        int64_t value() except+

    cdef _Time measure_microseconds() except+
    cdef _Time measure_nanoseconds() except+


cdef class PyTime(Unconstructable):
    cdef _Time _time

    cpdef int64_t value(self)

cpdef PyTime time(str units = *)
