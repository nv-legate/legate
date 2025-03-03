# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int32_t
from libcpp cimport bool
from libcpp.string cimport string as std_string

from ..utilities.tuple cimport _tuple
from ..utilities.unconstructable cimport Unconstructable


cdef extern from "legate/operation/projection.h" namespace "legate" nogil:
    cdef cppclass _SymbolicExpr "legate::SymbolicExpr":
        _SymbolicExpr() except+
        _SymbolicExpr(int32_t) except+
        _SymbolicExpr(int32_t, int32_t) except+
        _SymbolicExpr(int32_t, int32_t, int32_t) except+
        int32_t dim() except+
        int32_t weight() except+
        int32_t offset() except+
        bool is_identity(int32_t) except+
        bool operator==(const _SymbolicExpr&) except+
        _SymbolicExpr operator*(int32_t other) except+
        _SymbolicExpr operator+(int32_t other) except+
        std_string to_string() except+

    cdef _SymbolicExpr _dimension "legate::dimension" (int32_t) except+
    cdef _SymbolicExpr _constant "legate::constant" (int32_t) except+


ctypedef _tuple[_SymbolicExpr] _SymbolicPoint


cdef class SymbolicExpr(Unconstructable):
    cdef _SymbolicExpr _handle

    @staticmethod
    cdef SymbolicExpr from_handle(_SymbolicExpr)
    cpdef bool is_identity(self, int32_t dim)

cpdef SymbolicExpr dimension(int32_t dim)

cpdef SymbolicExpr constant(int32_t value)
