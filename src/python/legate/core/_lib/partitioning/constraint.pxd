# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
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
from ..utilities.unconstructable cimport Unconstructable
from .proxy cimport _ProxyConstraint

from collections.abc import Iterable


cdef extern from "legate/partitioning/constraint.h" namespace "legate" nogil:
    cdef cppclass _Variable "legate::Variable":
        _Variable() except+
        _Variable(const _Variable&) except+
        std_string to_string() except+

    cdef cppclass _Constraint "legate::Constraint":
        _Constraint() except+
        _Constraint(const _Constraint&) except+
        std_string to_string() except+

    cdef _Constraint _align "align" (_Variable, _Variable) except+

    # Using ... as argument list since Cython does not understand std::variant.
    cdef _ProxyConstraint _proxy_align "legate::align" (...) except+

    cdef _Constraint _broadcast "broadcast" (_Variable) except+

    cdef _Constraint _broadcast "broadcast" (
        _Variable, _tuple[uint32_t]
    ) except+

    # Using ... as argument list since Cython does not understand std::variant.
    cdef _ProxyConstraint _proxy_broadcast "broadcast" (...) except+

    cpdef enum class ImageComputationHint:
        NO_HINT
        MIN_MAX
        FIRST_LAST

    cdef _Constraint _image "image" (
        _Variable, _Variable, ImageComputationHint
    ) except+

    # Using ... as argument list since Cython does not understand std::variant.
    cdef _ProxyConstraint _proxy_image "image" (...) except+

    cdef _Constraint _scale "scale" (
        _tuple[uint64_t], _Variable, _Variable
    ) except+

    # Using ... as argument list since Cython does not understand std::variant.
    cdef _ProxyConstraint _proxy_scale "scale" (...) except+

    cdef _Constraint _bloat "bloat" (
        _Variable, _Variable, _tuple[uint64_t], _tuple[uint64_t]
    ) except+

    # Using ... as argument list since Cython does not understand std::variant.
    cdef _ProxyConstraint _proxy_bloat "bloat" (...) except+


cdef class Variable(Unconstructable):
    cdef _Variable _handle

    @staticmethod
    cdef Variable from_handle(_Variable)


cdef class Constraint(Unconstructable):
    cdef _Constraint _handle

    @staticmethod
    cdef Constraint from_handle(_Constraint)

cdef class DeferredConstraint:
    cdef:
        # The proxy function is considered an implementation detail and is not
        # exposed to Python
        _ProxyConstraint(*func)(tuple, tuple)
    cdef readonly:
        tuple[Any, ...] args

    @staticmethod
    cdef DeferredConstraint construct(
        _ProxyConstraint(*func)(tuple, tuple), tuple args
    )

ctypedef fused VariableOrStr:
    Variable
    str


cpdef object align(VariableOrStr lhs, VariableOrStr rhs)
cpdef object broadcast(VariableOrStr variable, axes: Iterable[int] =*)
cpdef object image(
    VariableOrStr var_function,
    VariableOrStr var_range,
    ImageComputationHint hint =*
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
