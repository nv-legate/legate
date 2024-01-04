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

from libc.stdint cimport int32_t, uint32_t, uintptr_t
from libcpp cimport bool
from libcpp.utility cimport move as std_move
from libcpp.vector cimport vector as std_vector

from enum import IntEnum, unique

import numpy as np

from ..legate_c cimport legate_core_reduction_op_kind_t


@unique
class ReductionOp(IntEnum):
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
    _Type.Code.BOOL : np.dtype(np.bool_),
    _Type.Code.INT8 : np.dtype(np.int8),
    _Type.Code.INT16 : np.dtype(np.int16),
    _Type.Code.INT32 : np.dtype(np.int32),
    _Type.Code.INT64 : np.dtype(np.int64),
    _Type.Code.UINT8 : np.dtype(np.uint8),
    _Type.Code.UINT16 : np.dtype(np.uint16),
    _Type.Code.UINT32 : np.dtype(np.uint32),
    _Type.Code.UINT64 : np.dtype(np.uint64),
    _Type.Code.FLOAT16 : np.dtype(np.float16),
    _Type.Code.FLOAT32 : np.dtype(np.float32),
    _Type.Code.FLOAT64 : np.dtype(np.float64),
    _Type.Code.COMPLEX64 : np.dtype(np.complex64),
    _Type.Code.COMPLEX128 : np.dtype(np.complex128),
    _Type.Code.STRING : np.dtype(np.str_),
}


cdef class Type:
    @staticmethod
    cdef Type from_handle(_Type ty):
        cdef Type result
        if ty.code() == _Type.Code.FIXED_ARRAY:
            result = FixedArrayType.__new__(FixedArrayType)
        elif ty.code() == _Type.Code.STRUCT:
            result = StructType.__new__(StructType)
        else:
            result = Type.__new__(Type)
        result._handle = ty
        return result

    def __init__(self) -> None:
        self._handle = _null_type()

    @property
    def code(self) -> int32_t:
        return <int32_t> self._handle.code()

    @property
    def size(self) -> uint32_t:
        return self._handle.size()

    @property
    def alignment(self) -> uint32_t:
        return self._handle.alignment()

    @property
    def uid(self) -> int32_t:
        return self._handle.uid()

    @property
    def variable_size(self) -> bool:
        return self._handle.variable_size()

    @property
    def is_primitive(self) -> bool:
        return self._handle.is_primitive()

    def record_reduction_op(
        self, int32_t op_kind, int64_t reduction_op_id
    ) -> None:
        self._handle.record_reduction_operator(op_kind, reduction_op_id)

    def reduction_op_id(self, int32_t op_kind) -> int64_t:
        return self._handle.find_reduction_operator(op_kind)

    def __repr__(self) -> str:
        return self._handle.to_string().decode()

    def to_numpy_dtype(self):
        code = self.code
        if code in _NUMPY_DTYPES:
            return _NUMPY_DTYPES[self.code]
        else:
            raise ValueError(f"Invalid type code: {code}")

    @property
    def raw_ptr(self) -> uintptr_t:
        return <uintptr_t>(&self._handle)

    def __hash__(self) -> int:
        return hash((self.__class__, self.code))

    def __eq__(self, Type other) -> bool:
        return isinstance(other, Type) and self._handle == other._handle


cdef class FixedArrayType(Type):
    def __init__(self) -> None:
        raise ValueError(
            f"{type(self).__name__} objects must not be constructed directly"
        )

    @property
    def num_elements(self) -> uint32_t:
        return self._handle.as_fixed_array_type().num_elements()

    @property
    def element_type(self) -> Type:
        return Type.from_handle(
            self._handle.as_fixed_array_type().element_type()
        )

    def to_numpy_dtype(self):
        elem_ty = self.element_type.to_numpy_dtype()
        N = self.num_elements
        return np.dtype((elem_ty, (N, ))) if N > 1 else elem_ty


cdef class StructType(Type):
    def __init__(self) -> None:
        raise ValueError(
            f"{type(self).__name__} objects must not be constructed directly"
        )

    @property
    def num_fields(self) -> uint32_t:
        return self._handle.as_struct_type().num_fields()

    def field_type(self, uint32_t field_idx) -> Type:
        return Type.from_handle(
            self._handle.as_struct_type().field_type(field_idx)
        )

    @property
    def aligned(self) -> bool:
        return self._handle.as_struct_type().aligned()

    @property
    def offsets(self) -> tuple[uint32_t, ...]:
        return tuple(self._handle.as_struct_type().offsets())

    def to_numpy_dtype(self):
        num_fields = self.num_fields
        names = tuple(
            f"_{field_idx}" for field_idx in range(num_fields)
        )
        formats = tuple(
            self.field_type(field_idx).to_numpy_dtype()
            for field_idx in range(num_fields)
        )
        return np.dtype(
            {"names": names, "formats": formats}, align=self.aligned
        )

null_type = Type.from_handle(_null_type())
bool_ = Type.from_handle(_bool())
int8 = Type.from_handle(_int8())
int16 = Type.from_handle(_int16())
int32 = Type.from_handle(_int32())
int64 = Type.from_handle(_int64())
uint8 = Type.from_handle(_uint8())
uint16 = Type.from_handle(_uint16())
uint32 = Type.from_handle(_uint32())
uint64 = Type.from_handle(_uint64())
float16 = Type.from_handle(_float16())
float32 = Type.from_handle(_float32())
float64 = Type.from_handle(_float64())
complex64 = Type.from_handle(_complex64())
complex128 = Type.from_handle(_complex128())
string_type = Type.from_handle(_string_type())


def binary_type(uint32_t size) -> Type:
    return Type.from_handle(_binary_type(size))


def array_type(
    Type element_type, uint32_t N
) -> FixedArrayType:
    return <FixedArrayType> Type.from_handle(
        _fixed_array_type(element_type._handle, N)
    )


def struct_type(list field_types, bool align = True) -> StructType:
    cdef std_vector[_Type] types = std_vector[_Type]()
    for field_type in field_types:
        types.push_back(
            (<Type> field_type)._handle
        )
    return <StructType> Type.from_handle(
        _struct_type(std_move(types), align)
    )


def point_type(int32_t ndim) -> Type:
    return Type.from_handle(_point_type(ndim))


def rect_type(int32_t ndim) -> Type:
    return Type.from_handle(_rect_type(ndim))
