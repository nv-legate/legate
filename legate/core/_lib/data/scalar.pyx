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
    uintptr_t,
)
from libcpp cimport bool as cpp_bool
from libcpp.complex cimport complex as std_complex
from libcpp.string cimport string as std_string

from typing import Any

import numpy as np

from ..type.type_info cimport Type, _Type

from ..type.type_info import null_type

from ..legate_c cimport __convert_halfint_to_float, __half
from ..utilities.utils cimport is_iterable


cpdef void from_null(Scalar scalar, object _, Type __):
    scalar._handle = _Scalar()


cdef void from_buffer(Scalar scalar, object value, Type dtype):
    if not (isinstance(value, (bytes, np.ndarray))):
        raise ValueError(
            f"Expected bytes or NumPy ndarray, but got {type(value)}"
        )
    if not isinstance(value, bytes):
        arr = np.asarray(value, dtype=dtype.to_numpy_dtype())
        value = arr.tobytes()

    if len(value) != dtype.size:
        raise ValueError(
            f"Type {dtype} expects a value of size {dtype.size}, "
            f"but the size of value is {len(value)}"
        )
    scalar._handle = _Scalar(dtype._handle, <const char*>value, True)


cpdef void from_bool(Scalar scalar, object value, Type dtype):
    if isinstance(value, bool):
        scalar._handle = _Scalar(<cpp_bool> value, dtype._handle)
    else:
        from_buffer(scalar, value, dtype)


cpdef void from_int8(Scalar scalar, object value, Type dtype):
    try:
        scalar._handle = _Scalar(<int8_t> value, dtype._handle)
    except TypeError:
        from_buffer(scalar, value, dtype)


cpdef void from_int16(Scalar scalar, object value, Type dtype):
    try:
        scalar._handle = _Scalar(<int16_t> value, dtype._handle)
    except TypeError:
        from_buffer(scalar, value, dtype)


cpdef void from_int32(Scalar scalar, object value, Type dtype):
    try:
        scalar._handle = _Scalar(<int32_t> value, dtype._handle)
    except TypeError:
        from_buffer(scalar, value, dtype)


cpdef void from_int64(Scalar scalar, object value, Type dtype):
    try:
        scalar._handle = _Scalar(<int64_t> value, dtype._handle)
    except TypeError:
        from_buffer(scalar, value, dtype)


cpdef void from_uint8(Scalar scalar, object value, Type dtype):
    try:
        scalar._handle = _Scalar(<uint8_t> value, dtype._handle)
    except TypeError:
        from_buffer(scalar, value, dtype)


cpdef void from_uint16(Scalar scalar, object value, Type dtype):
    try:
        scalar._handle = _Scalar(<uint16_t> value, dtype._handle)
    except TypeError:
        from_buffer(scalar, value, dtype)


cpdef void from_uint32(Scalar scalar, object value, Type dtype):
    try:
        scalar._handle = _Scalar(<uint32_t> value, dtype._handle)
    except TypeError:
        from_buffer(scalar, value, dtype)


cpdef void from_uint64(Scalar scalar, object value, Type dtype):
    try:
        scalar._handle = _Scalar(<uint64_t> value, dtype._handle)
    except TypeError:
        from_buffer(scalar, value, dtype)


cpdef void from_float16(Scalar scalar, object value, Type dtype):
    try:
        scalar._handle = _Scalar(__half(<float> value), dtype._handle)
    except TypeError:
        from_buffer(scalar, value, dtype)


cpdef void from_float32(Scalar scalar, object value, Type dtype):
    try:
        scalar._handle = _Scalar(<float> value, dtype._handle)
    except TypeError:
        from_buffer(scalar, value, dtype)


cpdef void from_float64(Scalar scalar, object value, Type dtype):
    try:
        scalar._handle = _Scalar(<double> value, dtype._handle)
    except TypeError:
        from_buffer(scalar, value, dtype)


cpdef void from_complex64(Scalar scalar, object value, Type dtype):
    try:
        scalar._handle = _Scalar(<std_complex[float]> value, dtype._handle)
    except TypeError:
        from_buffer(scalar, value, dtype)


cpdef void from_complex128(
    Scalar scalar, object value, Type dtype
):
    try:
        scalar._handle = _Scalar(<std_complex[double]> value, dtype._handle)
    except TypeError:
        from_buffer(scalar, value, dtype)


cpdef void from_array(Scalar scalar, object value, Type dtype):
    if not (isinstance(value, (bytes, np.ndarray)) or is_iterable(value)):
        raise ValueError(
            "Expected bytes, NumPy ndarray, or iterable, "
            f"but got {type(value)}"
        )
    if not isinstance(value, bytes):
        elem_type = dtype.element_type.to_numpy_dtype()
        arr = np.asarray(value, dtype=elem_type)
        value = arr.tobytes()

    if len(value) != dtype.size:
        raise ValueError(
            f"Type {dtype} expects a value of size {dtype.size}, "
            f"but the size of value is {len(value)}"
        )
    scalar._handle = _Scalar(dtype._handle, <const char*>value, True)


cpdef void from_struct(Scalar scalar, object value, Type dtype):
    if not isinstance(value, (bytes, tuple, np.ndarray)):
        raise ValueError(
            "Expected bytes, NumPy ndarray, or tuple, "
            f"but got {type(value)}"
        )
    if not isinstance(value, bytes):
        arr = np.asarray(value, dtype=dtype.to_numpy_dtype())
        value = arr.tobytes()

    if len(value) != dtype.size:
        raise ValueError(
            f"Type {dtype} expects a value of size {dtype.size}, "
            f"but the size of value is {len(value)}"
        )
    scalar._handle = _Scalar(dtype._handle, <const char*>value, True)


cpdef void from_string(Scalar scalar, str value, Type _):
    scalar._handle = _Scalar(<std_string> value.encode())


cdef dict _CONSTRUCTORS = {
    _Type.Code.NIL  : from_null,
    _Type.Code.BOOL: from_bool,
    _Type.Code.INT8: from_int8,
    _Type.Code.INT16: from_int16,
    _Type.Code.INT32: from_int32,
    _Type.Code.INT64: from_int64,
    _Type.Code.UINT8: from_uint8,
    _Type.Code.UINT16: from_uint16,
    _Type.Code.UINT32: from_uint32,
    _Type.Code.UINT64: from_uint64,
    _Type.Code.FLOAT16 : from_float16,
    _Type.Code.FLOAT32: from_float32,
    _Type.Code.FLOAT64: from_float64,
    _Type.Code.COMPLEX64: from_complex64,
    _Type.Code.COMPLEX128: from_complex128,
    _Type.Code.FIXED_ARRAY: from_array,
    _Type.Code.STRUCT: from_struct,
    _Type.Code.STRING: from_string,
}


cpdef tuple to_struct(Scalar scalar):
    arr = np.asarray(scalar)
    v = arr[()]
    # Unfortunately, the array interface requires paddings in a struct type to
    # be materialized as separate fields, so we need to filter out garbage
    # values from those paddings.
    cdef list result = []
    cdef int idx
    cdef str name
    for idx, (name, _) in enumerate(arr.dtype.descr):
        if name.startswith("_"):
            result.append(v[idx])
    return tuple(result)


cdef dict _GETTERS = {
    _Type.Code.NIL  : lambda _: None,
    _Type.Code.BOOL  : lambda Scalar result: result._handle.value[cpp_bool](),
    _Type.Code.INT8  : lambda Scalar result: result._handle.value[int8_t](),
    _Type.Code.INT16 : lambda Scalar result: result._handle.value[int16_t](),
    _Type.Code.INT32 : lambda Scalar result: result._handle.value[int32_t](),
    _Type.Code.INT64 : lambda Scalar result: result._handle.value[int64_t](),
    _Type.Code.UINT8  : lambda Scalar result: result._handle.value[uint8_t](),
    _Type.Code.UINT16 : lambda Scalar result: result._handle.value[uint16_t](),
    _Type.Code.UINT32 : lambda Scalar result: result._handle.value[uint32_t](),
    _Type.Code.UINT64 : lambda Scalar result: result._handle.value[uint64_t](),
    # We have to go through this song and dance because:
    #
    # 1. Cython cannot convert the C++ __half class to a python object (lambdas
    #    cannot return non-Python types)
    # 2. Even though __half has a operator float() (and implicit conversion to
    #    it), Cython (3.0.3) does not yet support defining it for C++ classes.
    _Type.Code.FLOAT16 : lambda Scalar result: __convert_halfint_to_float(
        result._handle.value[__half]().raw()
    ),
    _Type.Code.FLOAT32 : lambda Scalar result: result._handle.value[float](),
    _Type.Code.FLOAT64 : lambda Scalar result: result._handle.value[double](),
    _Type.Code.COMPLEX64 : lambda Scalar result: complex(
        result._handle.value[std_complex[float]]()
    ),
    _Type.Code.COMPLEX128 : lambda Scalar result: complex(
        result._handle.value[std_complex[double]]()
    ),
    _Type.Code.FIXED_ARRAY: np.asarray,
    _Type.Code.STRUCT: to_struct,
    _Type.Code.STRING : lambda Scalar result: (
        result._handle.value[std_string]().decode()
    )
}


cdef class Scalar:
    @staticmethod
    cdef Scalar from_handle(_Scalar handle):
        cdef Scalar result = Scalar.__new__(Scalar)
        result._handle = handle
        return result

    def __init__(self, value: Any, Type dtype) -> None:
        ctor = _CONSTRUCTORS.get(dtype.code)
        if ctor is None:
            raise TypeError(f"unhandled type {dtype}")
        ctor(self, value, dtype)

    cpdef object value(self):
        cdef _Type.Code code = self._handle.type().code()
        try:
            return _GETTERS[code](self)
        except KeyError as ke:
            raise RuntimeError(f"unhandled type {code}") from ke

    def __str__(self) -> str:
        return f"Scalar({self.value()}, {self.type})"

    def __repr__(self) -> str:
        return str(self)

    @property
    def __array_interface__(self):
        cdef Type ty = self.type
        if ty.variable_size:
            raise ValueError(
                "Scalars with variable size types don't support "
                "array interface"
            )
        if ty.code == _Type.Code.FIXED_ARRAY:
            shape = (ty.num_elements,)
            numpy_type = ty.element_type.to_numpy_dtype()
            return {
                "version": 3,
                "shape": shape,
                "typestr": numpy_type.str,
                "data": (<uintptr_t> self._handle.ptr(), True),
            }
        else:
            numpy_type = ty.to_numpy_dtype()
            return {
                "version": 3,
                "shape": (),
                "typestr": numpy_type.str,
                "data": (<uintptr_t> self._handle.ptr(), True),
                "descr": numpy_type.descr,
            }

    @property
    def type(self) -> Type:
        return Type.from_handle(self._handle.type())

    @property
    def ptr(self) -> uintptr_t:
        return <uintptr_t> self._handle.ptr()

    @property
    def raw_handle(self) -> uintptr_t:
        return <uintptr_t> &self._handle

    @staticmethod
    def null() -> Scalar:
        return Scalar(None, null_type)
