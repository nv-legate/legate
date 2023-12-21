# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from libc.stdint cimport int32_t
from libcpp cimport bool
from libcpp.string cimport string as std_string

from ..utilities.tuple cimport tuple as _tuple


cdef extern from "core/operation/projection.h" namespace "legate" nogil:
    cdef cppclass _SymbolicExpr "legate::SymbolicExpr":
        _SymbolicExpr()
        _SymbolicExpr(int32_t)
        _SymbolicExpr(int32_t, int32_t)
        _SymbolicExpr(int32_t, int32_t, int32_t)
        int32_t dim() const
        int32_t weight() const
        int32_t offset() const
        bool is_identity(int32_t) const
        bool operator==(const _SymbolicExpr&) const
        _SymbolicExpr operator*(int32_t other) const
        _SymbolicExpr operator+(int32_t other) const
        std_string to_string() const

    cdef _SymbolicExpr _dimension "legate::dimension" (int32_t)
    cdef _SymbolicExpr _constant "legate::constant" (int32_t)


ctypedef _tuple[_SymbolicExpr] _SymbolicPoint


cdef class SymbolicExpr:
    cdef _SymbolicExpr _handle

    @staticmethod
    cdef SymbolicExpr from_handle(_SymbolicExpr)
    cpdef bool is_identity(self, int32_t dim)

cpdef SymbolicExpr dimension(int32_t dim)

cpdef SymbolicExpr constant(int32_t value)
