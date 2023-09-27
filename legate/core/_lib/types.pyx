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

from libcpp cimport bool
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector

import numpy as np


cdef extern from "core/legate_c.h" nogil:
    ctypedef enum legate_core_type_code_t:
        _NULL "NULL_LT"
        _BOOL "BOOL_LT"
        _INT8 "INT8_LT"
        _INT16 "INT16_LT"
        _INT32 "INT32_LT"
        _INT64 "INT64_LT"
        _UINT8 "UINT8_LT"
        _UINT16 "UINT16_LT"
        _UINT32 "UINT32_LT"
        _UINT64 "UINT64_LT"
        _FLOAT16 "FLOAT16_LT"
        _FLOAT32 "FLOAT32_LT"
        _FLOAT64 "FLOAT64_LT"
        _COMPLEX64 "COMPLEX64_LT"
        _COMPLEX128 "COMPLEX128_LT"
        _BINARY "BINARY_LT"
        _FIXED_ARRAY "FIXED_ARRAY_LT"
        _STRUCT "STRUCT_LT"
        _STRING "STRING_LT"

    ctypedef enum legate_core_reduction_op_kind_t:
        _ADD "ADD_LT"
        _SUB "SUB_LT"
        _MUL "MUL_LT"
        _DIV "DIV_LT"
        _MAX "MAX_LT"
        _MIN "MIN_LT"
        _OR  "OR_LT"
        _AND "AND_LT"
        _XOR "XOR_LT"

NIL = legate_core_type_code_t._NULL
BOOL = legate_core_type_code_t._BOOL
INT8 = legate_core_type_code_t._INT8
INT16 = legate_core_type_code_t._INT16
INT32 = legate_core_type_code_t._INT32
INT64 = legate_core_type_code_t._INT64
UINT8 = legate_core_type_code_t._UINT8
UINT16 = legate_core_type_code_t._UINT16
UINT32 = legate_core_type_code_t._UINT32
UINT64 = legate_core_type_code_t._UINT64
FLOAT16 = legate_core_type_code_t._FLOAT16
FLOAT32 = legate_core_type_code_t._FLOAT32
FLOAT64 = legate_core_type_code_t._FLOAT64
COMPLEX64 = legate_core_type_code_t._COMPLEX64
COMPLEX128 = legate_core_type_code_t._COMPLEX128
BINARY = legate_core_type_code_t._BINARY
FIXED_ARRAY = legate_core_type_code_t._FIXED_ARRAY
STRUCT = legate_core_type_code_t._STRUCT
STRING = legate_core_type_code_t._STRING

ADD = legate_core_reduction_op_kind_t._ADD
SUB = legate_core_reduction_op_kind_t._SUB
MUL = legate_core_reduction_op_kind_t._MUL
DIV = legate_core_reduction_op_kind_t._DIV
MAX = legate_core_reduction_op_kind_t._MAX
MIN = legate_core_reduction_op_kind_t._MIN
OR = legate_core_reduction_op_kind_t._OR
AND = legate_core_reduction_op_kind_t._AND
XOR = legate_core_reduction_op_kind_t._XOR

_NUMPY_DTYPES = {
    BOOL : np.dtype(np.bool_),
    INT8 : np.dtype(np.int8),
    INT16 : np.dtype(np.int16),
    INT32 : np.dtype(np.int32),
    INT64 : np.dtype(np.int64),
    UINT8 : np.dtype(np.uint8),
    UINT16 : np.dtype(np.uint16),
    UINT32 : np.dtype(np.uint32),
    UINT64 : np.dtype(np.uint64),
    FLOAT16 : np.dtype(np.float16),
    FLOAT32 : np.dtype(np.float32),
    FLOAT64 : np.dtype(np.float64),
    COMPLEX64 : np.dtype(np.complex64),
    COMPLEX128 : np.dtype(np.complex128),
    STRING : np.dtype(np.str_),
}


cdef extern from "core/type/detail/type_info.h" namespace "legate::detail" nogil:
    cdef cppclass Type:
        ctypedef enum Code:
            pass
        int code
        unsigned int size() except+
        unsigned int alignment()
        int uid()
        bool variable_size()
        string to_string()
        bool is_primitive()
        void record_reduction_operator(int, int) except+
        int find_reduction_operator(int) except+

    cdef cppclass FixedArrayType(Type):
        unsigned int num_elements()
        shared_ptr[Type] element_type()

    cdef cppclass StructType(Type):
        unsigned int num_fields()
        shared_ptr[Type] field_type(unsigned int)
        bool aligned()

    cdef shared_ptr[Type] primitive_type(int code)

    cdef shared_ptr[Type] string_type()

    cdef shared_ptr[Type] binary_type(unsigned int size)

    cdef shared_ptr[Type] fixed_array_type(
        shared_ptr[Type] element_type, unsigned int N
    ) except+

    cdef shared_ptr[Type] struct_type(
        vector[shared_ptr[Type]] field_types, bool
    ) except+


cdef Dtype from_ptr(shared_ptr[Type] ty):
    cdef Dtype dtype
    if <int> ty.get().code == FIXED_ARRAY:
        dtype = FixedArrayDtype.__new__(FixedArrayDtype)
    elif <int> ty.get().code == STRUCT:
        dtype = StructDtype.__new__(StructDtype)
    else:
        dtype = Dtype.__new__(Dtype)
    dtype._type = move(ty)
    return dtype


cdef class Dtype:
    cdef shared_ptr[Type] _type

    @staticmethod
    def primitive_type(int code) -> Dtype:
        return from_ptr(primitive_type(<Type.Code> code))

    @staticmethod
    def string_type() -> Dtype:
        return from_ptr(string_type())

    @staticmethod
    def binary_type(unsigned size) -> Dtype:
        return from_ptr(binary_type(size))

    @staticmethod
    def fixed_array_type(
        Dtype element_type, unsigned N
    ) -> FixedArrayDtype:
        return <FixedArrayDtype> from_ptr(
            fixed_array_type(element_type._type, N)
        )

    @staticmethod
    def struct_type(list field_types, bool align) -> StructDtype:
        cdef vector[shared_ptr[Type]] types
        for field_type in field_types:
            types.push_back(
                (<Dtype> field_type)._type
            )
        return <StructDtype> from_ptr(
            struct_type(move(types), align)
        )

    @property
    def code(self) -> int:
        return <int> self._type.get().code

    @property
    def size(self) -> int:
        return self._type.get().size()

    @property
    def alignment(self) -> int:
        return self._type.get().alignment()

    @property
    def uid(self) -> int:
        return self._type.get().uid()

    @property
    def variable_size(self) -> bool:
        return self._type.get().variable_size()

    @property
    def is_primitive(self) -> bool:
        return self._type.get().is_primitive()

    def record_reduction_op(self, int op_kind, int reduction_op_id) -> None:
        self._type.get().record_reduction_operator(op_kind, reduction_op_id)

    def reduction_op_id(self, int op_kind) -> int:
        return self._type.get().find_reduction_operator(op_kind)

    def __repr__(self) -> str:
        return self._type.get().to_string().decode()

    def to_numpy_dtype(self):
        code = self.code
        if code in _NUMPY_DTYPES:
            return _NUMPY_DTYPES[self.code]
        else:
            raise ValueError(f"Invalid type code: {code}")

    def serialize(self, buf) -> None:
        buf.pack_32bit_int(self.code)

    @property
    def raw_ptr(self) -> long:
        return <long>(self._type.get())


cdef class FixedArrayDtype(Dtype):
    def num_elements(self) -> int:
        cdef FixedArrayType* ty = <FixedArrayType*> self._type.get()
        return ty.num_elements()

    @property
    def element_type(self) -> Dtype:
        cdef FixedArrayType* ty = <FixedArrayType*> self._type.get()
        return from_ptr(ty.element_type())

    def to_numpy_dtype(self):
        arr_type = (
            self.element_type.to_numpy_dtype(), self.num_elements()
        )
        # Return a singleton struct type, as NumPy would flatten away
        # nested arrays
        return np.dtype({"names": ("_0",), "formats": (arr_type,)})

    def serialize(self, buf) -> None:
        buf.pack_32bit_int(self.code)
        buf.pack_32bit_int(self.uid)
        buf.pack_32bit_uint(self.num_elements())
        self.element_type.serialize(buf)


cdef class StructDtype(Dtype):
    def num_fields(self) -> int:
        cdef StructType* ty = <StructType*> self._type.get()
        return ty.num_fields()

    def field_type(self, int field_idx) -> Dtype:
        cdef StructType* ty = <StructType*> self._type.get()
        return from_ptr(ty.field_type(field_idx))

    def aligned(self) -> bool:
        cdef StructType* ty = <StructType*> self._type.get()
        return ty.aligned()

    def to_numpy_dtype(self):
        num_fields = self.num_fields()
        names = tuple(
            f"_{field_idx}" for field_idx in range(num_fields)
        )
        formats = tuple(
            self.field_type(field_idx).to_numpy_dtype()
            for field_idx in range(num_fields)
        )
        return np.dtype(
            {"names": names, "formats": formats}, align=self.aligned()
        )

    def serialize(self, buf) -> None:
        buf.pack_32bit_int(self.code)
        num_fields = self.num_fields()
        buf.pack_32bit_int(self.uid)
        buf.pack_32bit_uint(num_fields)
        for field_idx in range(num_fields):
            self.field_type(field_idx).serialize(buf)
        buf.pack_bool(self.aligned)
