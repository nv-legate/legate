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
from libcpp.string cimport string as std_string

from ..data.shape cimport _Shape
from ..utilities.tuple cimport _tuple


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
        _Variable, const _tuple[int32_t]&
    ) except+

    cdef _Constraint _image "image" (_Variable, _Variable)

    cdef _Constraint _scale "scale" (const _Shape&, _Variable, _Variable)

    cdef _Constraint _bloat "bloat" (
        _Variable, _Variable, const _Shape&, const _Shape&
    )


cdef class Variable:
    cdef _Variable _handle

    @staticmethod
    cdef Variable from_handle(_Variable)


cdef class Constraint:
    cdef _Constraint _handle

    @staticmethod
    cdef Constraint from_handle(_Constraint)
