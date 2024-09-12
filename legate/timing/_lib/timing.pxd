# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from libc.stdint cimport int64_t

from ..._lib.utilities.unconstructable cimport Unconstructable


cdef extern from "legate/timing/timing.h" namespace "legate::timing" nogil:
    cdef cppclass Time:
        int64_t value()

    cdef Time measure_microseconds()
    cdef Time measure_nanoseconds()


cdef class PyTime(Unconstructable):
    cdef Time _time

    cpdef int64_t value(self)

cpdef PyTime time(str units = *)
