# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int32_t, uint32_t
from libcpp cimport bool
from libcpp.string cimport string as std_string
from libcpp.vector cimport vector as std_vector


# Cannot do:
#
# from ..utilities.typedefs cimport _GlobalRedopID
#
# Because otherwise python crashes on startup with weird errors such as:
#
# - Fatal Python error: Segmentation fault
# - KeyError: '__reduce_cython__'
# - scalar.pyx:37: in init legate.core._lib.data.scalar, ImportError: cannot
#   import name null_type
#
# My best guess is that utilities.typedefs introduces a circular import, which
# Cython seemingly does not handle properly. So we must unfortunately declare
# this type anew here...
cdef extern from "legate/utilities/typedefs.h" namespace "legate" nogil:
    ctypedef int _Legion_ReductionOpID "Legion::ReductionOpID"

    cdef enum class _GlobalRedopID "legate::GlobalRedopID" (
        _Legion_ReductionOpID
    ):
        pass

# Yes this is a hack. No, you cannot get it to work any other way. Declaring
# the enum inline within Type leads to confusing
#
#  error: use of undeclared identifier '__pyx_e_4Type_INT8'
# __pyx_t_3 = __Pyx_PyInt_From_legate_3a__3a_Type_3a__3a_Code(__pyx_e_4Type_INT8);  # noqa E501
#
# because cython mangles the name of the enum
cdef extern from "legate/type/types.h" namespace "legate" nogil:
    cpdef enum class TypeCode "legate::Type::Code":
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


cdef extern from "legate/type/types.h" namespace "legate" nogil:
    cpdef enum class ReductionOpKind:
        ADD
        MUL
        MAX
        MIN
        OR
        AND
        XOR

    cdef cppclass _Type "legate::Type":
        ctypedef TypeCode Code
        _Type.Code code() except+
        uint32_t size() except+
        uint32_t alignment() except+
        int32_t uid() except+
        bool variable_size() except+
        std_string to_string() except+
        bool is_primitive() except+
        _FixedArrayType as_fixed_array_type() except+
        _StructType as_struct_type() except+
        void record_reduction_operator(int32_t, _GlobalRedopID) except+
        _GlobalRedopID find_reduction_operator(int32_t) except+
        bool operator==(const _Type&) except+

    cdef cppclass _FixedArrayType "legate::FixedArrayType" (_Type):
        uint32_t num_elements() except+
        _Type element_type() except+

    cdef cppclass _StructType "legate::StructType" (_Type):
        uint32_t num_fields() except+
        _Type field_type(uint32_t) except+
        bool aligned() except+
        std_vector[uint32_t] offsets() except+

    cdef _Type _string_type "legate::string_type" () except+

    cdef _Type _binary_type "legate::binary_type" (uint32_t size) except+

    cdef _Type _null_type "legate::null_type" () except+

    cdef _Type _bool "legate::bool_" () except+
    cdef _Type _int8 "legate::int8" () except+
    cdef _Type _int16 "legate::int16" () except+
    cdef _Type _int32 "legate::int32" () except+
    cdef _Type _int64 "legate::int64" () except+
    cdef _Type _uint8 "legate::uint8" () except+
    cdef _Type _uint16 "legate::uint16" () except+
    cdef _Type _uint32 "legate::uint32" () except+
    cdef _Type _uint64 "legate::uint64" () except+
    cdef _Type _float16 "legate::float16" () except+
    cdef _Type _float32 "legate::float32" () except+
    cdef _Type _float64 "legate::float64" () except+
    cdef _Type _complex64 "legate::complex64" () except+
    cdef _Type _complex128 "legate::complex128" () except+

    cdef _FixedArrayType _point_type "legate::point_type" (
        uint32_t ndim
    ) except+

    cdef _StructType _rect_type "legate::rect_type"(uint32_t ndim) except+

    cdef _FixedArrayType _fixed_array_type "legate::fixed_array_type" (
        _Type element_type, uint32_t N
    ) except+

    cdef _StructType _struct_type "legate::struct_type" (
        std_vector[_Type] field_types, bool
    ) except+


cdef class Type:
    cdef _Type _handle

    @staticmethod
    cdef Type from_handle(_Type)

    cpdef void record_reduction_op(
        self, int32_t op_kind, _GlobalRedopID reduction_op_id
    )
    cpdef _GlobalRedopID reduction_op_id(self, int32_t op_kind)
    cpdef object to_numpy_dtype(self)

    @staticmethod
    cdef Type from_py_object(object py_object)

    @staticmethod
    cdef Type from_np_dtype(object np_dtype)

cdef class FixedArrayType(Type):
    cpdef object to_numpy_dtype(self)

cdef class StructType(Type):
    cpdef Type field_type(self, uint32_t field_idx)
    cpdef object to_numpy_dtype(self)

cpdef Type binary_type(uint32_t size)
cpdef FixedArrayType array_type(Type element_type, uint32_t N)
cpdef StructType struct_type(list field_types, bool align = *)
cpdef FixedArrayType point_type(int32_t ndim)
cpdef StructType rect_type(int32_t ndim)
