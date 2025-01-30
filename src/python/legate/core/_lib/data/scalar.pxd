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

from libc.stdint cimport (
    int8_t,
    int16_t,
    int32_t,
    int64_t,
    uint8_t,
    uint16_t,
    uint32_t,
    uint64_t,
)
from libcpp cimport bool as cpp_bool
from libcpp.complex cimport complex as std_complex

from ..._ext.cython_libcpp.string_view cimport string_view as std_string_view
from ..type.types cimport _Type
from ..utilities.typedefs cimport __half


cdef extern from "legate/data/scalar.h" namespace "legate" nogil:
    cdef cppclass _Scalar "legate::Scalar":
        _Scalar() except+
        _Scalar(cpp_bool, _Type) except+
        _Scalar(int8_t, _Type) except+
        _Scalar(int16_t, _Type) except+
        _Scalar(int32_t, _Type) except+
        _Scalar(int64_t, _Type) except+
        _Scalar(uint8_t, _Type) except+
        _Scalar(uint16_t, _Type) except+
        _Scalar(uint32_t, _Type) except+
        _Scalar(uint64_t, _Type) except+
        _Scalar(__half, _Type) except+
        _Scalar(float, _Type) except+
        _Scalar(double, _Type) except+
        _Scalar(std_complex[float], _Type) except+
        _Scalar(std_complex[double], _Type) except+
        _Scalar(_Scalar) except+
        _Scalar(std_string_view) except+
        _Scalar(_Type, const char*, cpp_bool) except+
        _Type type() except+
        size_t size() except+
        T value[T]() except+
        const void* ptr() except+


cdef class Scalar:
    cdef _Scalar _handle

    @staticmethod
    cdef Scalar from_handle(_Scalar)

    cpdef object value(self)
