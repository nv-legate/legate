# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp.vector cimport vector as std_vector
from libc.stdint cimport int32_t

from ..utilities.unconstructable cimport Unconstructable

cdef extern from "legate/mapping/mapping.h" namespace "legate::mapping" nogil:
    cpdef enum class TaskTarget:
        GPU
        OMP
        CPU

    cpdef enum class StoreTarget:
        SYSMEM
        FBMEM
        ZCMEM
        SOCKETMEM

    cpdef enum class DimOrderingKind "legate::mapping::DimOrdering::Kind":
        C
        FORTRAN
        CUSTOM

    cdef cppclass _DimOrdering "legate::mapping::DimOrdering":
        @staticmethod
        _DimOrdering c_order() except+

        @staticmethod
        _DimOrdering fortran_order() except+

        @staticmethod
        _DimOrdering custom_order(std_vector[int32_t] dims) except+

        DimOrderingKind kind() except+

    cdef cppclass _Mapper "legate::mapping::Mapper":
        pass


cdef class DimOrdering(Unconstructable):
    cdef _DimOrdering _handle

    @staticmethod
    cdef DimOrdering from_handle(_DimOrdering handle)
