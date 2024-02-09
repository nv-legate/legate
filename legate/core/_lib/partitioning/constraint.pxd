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

from libc.stdint cimport uint32_t, uint64_t
from libcpp.string cimport string as std_string

from ..utilities.tuple cimport _tuple

from collections.abc import Iterable


cdef extern from "core/partitioning/constraint.h" namespace "legate" nogil:
    cdef cppclass _Variable "legate::Variable":
        _Variable()
        _Variable(const _Variable&)
        std_string to_string() const

    cdef cppclass _Constraint "legate::Constraint":
        _Constraint()
        _Constraint(const _Constraint&)
        std_string to_string() const

    cdef _Constraint _align "align" (_Variable, _Variable)

    cdef _Constraint _broadcast "broadcast" (_Variable)

    cdef _Constraint _broadcast "broadcast" (
        _Variable, _tuple[uint32_t]
    ) except+

    cdef _Constraint _image "image" (_Variable, _Variable)

    cdef _Constraint _scale "scale" (
        _tuple[uint64_t], _Variable, _Variable
    )

    cdef _Constraint _bloat "bloat" (
        _Variable, _Variable, _tuple[uint64_t], _tuple[uint64_t]
    )


cdef class Variable:
    cdef _Variable _handle

    @staticmethod
    cdef Variable from_handle(_Variable)


cdef class Constraint:
    cdef _Constraint _handle

    @staticmethod
    cdef Constraint from_handle(_Constraint)

cdef class ConstraintProxy:
    cdef readonly:
        object func
        tuple[Any, ...] args

ctypedef fused VariableOrStr:
    Variable
    str


cpdef object align(VariableOrStr lhs, VariableOrStr rhs)
cpdef object broadcast(VariableOrStr variable, axes: Iterable[int] =*)
cpdef object image(
    VariableOrStr var_function, VariableOrStr var_range
)
cpdef object scale(
    tuple factors,
    VariableOrStr var_smaller,
    VariableOrStr var_bigger
)
cpdef object bloat(
    VariableOrStr var_source,
    VariableOrStr var_bloat,
    tuple low_offsets,
    tuple high_offsets,
)
