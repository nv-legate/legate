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

from libc.stdint cimport int32_t, int64_t, uint32_t
from libcpp cimport bool
from libcpp.string cimport string as std_string
from libcpp.vector cimport vector as std_vector


# Yes this is a hack. No, you cannot get it to work any other way. Declaring
# the enum inline within Type leads to confusing
#
#  error: use of undeclared identifier '__pyx_e_4Type_INT8'
# __pyx_t_3 = __Pyx_PyInt_From_legate_3a__3a_Type_3a__3a_Code(__pyx_e_4Type_INT8);  # noqa E501
#
# because cython mangles the name of the enum
cdef extern from "core/type/type_info.h" namespace "legate::Type" nogil:
    cdef enum class Code:
        NIL
        BOOL
        INT8
        INT16
        INT32
        INT64
        UINT8
        UINT16
        UINT32
        UINT64
        FLOAT16
        FLOAT32
        FLOAT64
        COMPLEX64
        COMPLEX128
        BINARY
        FIXED_ARRAY
        STRUCT
        STRING
        LIST


cdef extern from "core/type/type_info.h" namespace "legate" nogil:
    cdef cppclass _Type "legate::Type":
        ctypedef Code Code
        _Type.Code code()
        uint32_t size() except+
        uint32_t alignment()
        int32_t uid()
        bool variable_size()
        std_string to_string()
        bool is_primitive()
        _FixedArrayType as_fixed_array_type()
        _StructType as_struct_type()
        void record_reduction_operator(int32_t, int64_t) except+
        int64_t find_reduction_operator(int32_t) except+
        bool operator==(const _Type&) const

    cdef cppclass _FixedArrayType(_Type):
        uint32_t num_elements()
        _Type element_type()

    cdef cppclass _StructType(_Type):
        uint32_t num_fields()
        _Type field_type(uint32_t)
        bool aligned()
        std_vector[uint32_t] offsets() const

    cdef _Type _string_type "legate::string_type" ()

    cdef _Type _binary_type "legate::binary_type" (uint32_t size)

    cdef _Type _null_type "legate::null_type" ()

    cdef _Type _bool "legate::bool_" ()
    cdef _Type _int8 "legate::int8" ()
    cdef _Type _int16 "legate::int16" ()
    cdef _Type _int32 "legate::int32" ()
    cdef _Type _int64 "legate::int64" ()
    cdef _Type _uint8 "legate::uint8" ()
    cdef _Type _uint16 "legate::uint16" ()
    cdef _Type _uint32 "legate::uint32" ()
    cdef _Type _uint64 "legate::uint64" ()
    cdef _Type _float16 "legate::float16" ()
    cdef _Type _float32 "legate::float32" ()
    cdef _Type _float64 "legate::float64" ()
    cdef _Type _complex64 "legate::complex64" ()
    cdef _Type _complex128 "legate::complex128" ()

    cdef _Type _point_type "legate::point_type"(int32_t ndim)

    cdef _Type _rect_type "legate::rect_type"(int32_t ndim)

    cdef _Type _fixed_array_type "legate::fixed_array_type" (
        _Type element_type, uint32_t N
    ) except+

    cdef _Type _struct_type "legate::struct_type" (
        std_vector[_Type] field_types, bool
    ) except+


cdef class Type:
    cdef _Type _handle

    @staticmethod
    cdef Type from_handle(_Type)
