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
from libcpp.string cimport string as std_string

from ..legate_c cimport __half
from ..type.type_info cimport _Type


cdef extern from "core/data/scalar.h" namespace "legate" nogil:
    cdef cppclass _Scalar "legate::Scalar":
        _Scalar()
        _Scalar(cpp_bool, _Type)
        _Scalar(int8_t, _Type)
        _Scalar(int16_t, _Type)
        _Scalar(int32_t, _Type)
        _Scalar(int64_t, _Type)
        _Scalar(uint8_t, _Type)
        _Scalar(uint16_t, _Type)
        _Scalar(uint32_t, _Type)
        _Scalar(uint64_t, _Type)
        _Scalar(__half, _Type)
        _Scalar(float, _Type)
        _Scalar(double, _Type)
        _Scalar(std_complex[float], _Type)
        _Scalar(std_complex[double], _Type)
        _Scalar(_Scalar)
        _Scalar(const std_string&)
        _Scalar(_Type, const char*, cpp_bool)
        _Type type() const
        size_t size() const
        T value[T]() except +
        const void* ptr() const


cdef class Scalar:
    cdef _Scalar _handle

    @staticmethod
    cdef Scalar from_handle(_Scalar)

    cpdef object value(self)
