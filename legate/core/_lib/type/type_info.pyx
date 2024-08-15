# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

import builtins

import numpy as np


cdef dict _TO_NUMPY_DTYPES = {
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

cdef dict _FROM_NUMPY_DTYPES = {
    np.dtype(np.bool_) : bool_,
    np.dtype(np.int8) : int8,
    np.dtype(np.int16) : int16,
    np.dtype(np.int32) : int32,
    np.dtype(np.int64) : int64,
    np.dtype(np.uint8) : uint8,
    np.dtype(np.uint16) : uint16,
    np.dtype(np.uint32) : uint32,
    np.dtype(np.uint64) : uint64,
    np.dtype(np.float16) : float16,
    np.dtype(np.float32) : float32,
    np.dtype(np.float64) : float64,
    np.dtype(np.complex64) : complex64,
    np.dtype(np.complex128) : complex128,
    np.dtype(np.str_) : string_type,
}

cdef inline Type deduce_numpy_type_(object py_object):
    try:
        return _FROM_NUMPY_DTYPES[py_object]
    except KeyError:
        raise NotImplementedError(f"Unhandled numpy data type: {py_object}")

cdef inline Type deduce_sized_scalar_type_(object py_object):
    # numpys conversion algorithm is overly pessimistic for floating point
    # values with a large number of significant decimal digits. This leads to
    # the unfortunate case that the returned dtypes do not round-trip back to
    # the same value. For example:
    #
    # >>> f = 261.3544313109841
    # >>> np.min_scalar_type(f)
    # dtype('float16')
    # >>> np.float16(f)
    # np.float16(261.2)
    # >>> float(np.float16(f))
    # 261.25
    #
    # So we use the largest possible floating point type to cover our bases.
    if isinstance(py_object, builtins.float):
        return float64
    if isinstance(py_object, builtins.complex):
        return complex128
    return deduce_numpy_type_(np.min_scalar_type(py_object))

cdef inline Type deduce_array_type_(object py_object):
    cdef uint32_t n_sub = len(py_object)

    if n_sub == 0:
        # We _should_ be able to do this if we know the type-hint that this
        # corresponds to, but that would require a rework of these deduction
        # methods to parse the type hint instead of the object itself.
        raise NotImplementedError(
            "Cannot yet deduce sub-array types from empty container: "
            f"{py_object} (of type {type(py_object)})"
        )

    cdef Type first_type = Type.from_py_object(py_object[0])
    cdef Type sub_type
    cdef int idx

    # Don't have to do this check if we're given a numpy array, because numpy
    # will ensure all elems have the same type
    if not isinstance(py_object, np.ndarray):
        for idx, ty in enumerate(py_object):
            if (sub_type := Type.from_py_object(ty)) != first_type:
                raise NotImplementedError(
                    f"Unsupported type: {py_object!r}. All elements must have "
                    f"the same type. Element at index {idx} has type "
                    f"{sub_type}, expected {first_type}"
                )
            if idx > 10:
                # fail-safe in case we get a large iterable, in which case we
                # simply trust that the user has done the right thing
                break

    return array_type(first_type, n_sub)

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
    def code(self) -> _Type.Code:
        return <_Type.Code>self._handle.code()

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

    cpdef void record_reduction_op(
        self, int32_t op_kind, _GlobalRedopID reduction_op_id
    ):
        self._handle.record_reduction_operator(op_kind, reduction_op_id)

    cpdef _GlobalRedopID reduction_op_id(self, int32_t op_kind):
        return self._handle.find_reduction_operator(op_kind)

    def __repr__(self) -> str:
        return self._handle.to_string().decode()

    cpdef object to_numpy_dtype(self):
        cdef _Type.Code code = self.code
        try:
            return _TO_NUMPY_DTYPES[code]
        except KeyError:
            raise ValueError(f"Invalid type code: {code}")

    @property
    def raw_ptr(self) -> uintptr_t:
        return <uintptr_t>(&self._handle)

    def __hash__(self) -> int:
        return hash((self.__class__, self.code))

    def __eq__(self, Type other) -> bool:
        return isinstance(other, Type) and self._handle == other._handle

    @staticmethod
    cdef Type from_py_object(object py_object):
        if isinstance(py_object, Type):
            return py_object
        if isinstance(py_object, builtins.bool):
            return bool_
        if isinstance(
            py_object, (builtins.int, builtins.float, builtins.complex)
        ):
            return deduce_sized_scalar_type_(py_object)
        if isinstance(py_object, builtins.str):
            return string_type
        if isinstance(py_object, (list, tuple, np.ndarray)):
            return deduce_array_type_(py_object)
        if isinstance(py_object, np.generic):
            return deduce_numpy_type_(py_object.dtype)
        raise NotImplementedError(f"unsupported type: {py_object!r}")


cdef class FixedArrayType(Type):
    # Cannot use Unconstructable here because Cython only allows 1 extension
    # type base-class
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

    cpdef object to_numpy_dtype(self):
        elem_ty = self.element_type.to_numpy_dtype()
        cdef uint32_t N = self.num_elements
        return np.dtype((elem_ty, (N, ))) if N > 1 else elem_ty


cdef class StructType(Type):
    # Cannot use Unconstructable here because Cython only allows 1 extension
    # type base-class
    def __init__(self) -> None:
        raise ValueError(
            f"{type(self).__name__} objects must not be constructed directly"
        )

    @property
    def num_fields(self) -> uint32_t:
        return self._handle.as_struct_type().num_fields()

    cpdef Type field_type(self, uint32_t field_idx):
        return Type.from_handle(
            self._handle.as_struct_type().field_type(field_idx)
        )

    @property
    def aligned(self) -> bool:
        return self._handle.as_struct_type().aligned()

    @property
    def offsets(self) -> tuple[uint32_t, ...]:
        return tuple(self._handle.as_struct_type().offsets())

    cpdef object to_numpy_dtype(self):
        cdef uint32_t num_fields = self.num_fields
        # Need to construct these tuple using list comprehensions because
        # Cython (as of 3.0.8) does not yet support closures in cdef or cpdef
        # functions.
        #
        # For whatever reason comprehensions are treated specially by the
        # cython compiler (even though they are also technically closures...).
        cdef tuple names = tuple(
            [f"_{field_idx}" for field_idx in range(num_fields)]
        )
        cdef tuple formats = tuple(
            [
                self.field_type(field_idx).to_numpy_dtype()
                for field_idx in range(num_fields)
            ]
        )
        return np.dtype(
            {"names": names, "formats": formats}, align=self.aligned
        )

cpdef Type binary_type(uint32_t size):
    return Type.from_handle(_binary_type(size))


cpdef FixedArrayType array_type(Type element_type, uint32_t N):
    return <FixedArrayType> Type.from_handle(
        _fixed_array_type(element_type._handle, N)
    )


cpdef StructType struct_type(list field_types, bool align = True):
    cdef std_vector[_Type] types = std_vector[_Type]()
    cdef Type field_type

    types.reserve(len(field_types))
    for field_type in field_types:
        types.push_back(
            (<Type> field_type)._handle
        )
    return <StructType> Type.from_handle(
        _struct_type(std_move(types), align)
    )


cpdef FixedArrayType point_type(int32_t ndim):
    return FixedArrayType.from_handle(_point_type(ndim))


cpdef StructType rect_type(int32_t ndim):
    return StructType.from_handle(_rect_type(ndim))
