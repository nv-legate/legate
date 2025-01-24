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

from cpython.buffer cimport PyBUF_READ
from cpython.memoryview cimport PyMemoryView_FromMemory
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
from libcpp.utility cimport move as std_move

from ..._ext.cython_libcpp.string_view cimport (
    str_from_string_view,
    string_view as std_string_view,
    string_view_from_py as std_string_view_from_py,
)
from ..type.type_info cimport Type

from typing import Any

import numpy as np

from ..type.type_info cimport Type, _Type

from ..type.type_info import null_type

from ..utilities.typedefs cimport __half, half_to_float
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

    from_binary(scalar, value, dtype)

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
    if isinstance(value, np.ndarray):
        value = value.tobytes()
    elif not isinstance(value, bytes):
        elem_type = dtype.element_type.to_numpy_dtype()
        arr = np.asarray(value, dtype=elem_type)
        value = arr.tobytes()

    from_binary(scalar, value, dtype)


cpdef void from_struct(Scalar scalar, object value, Type dtype):
    if not isinstance(value, (bytes, tuple, np.ndarray)):
        raise ValueError(
            "Expected bytes, NumPy ndarray, or tuple, "
            f"but got {type(value)}"
        )
    if not isinstance(value, bytes):
        arr = np.asarray(value, dtype=dtype.to_numpy_dtype())
        value = arr.tobytes()

    from_binary(scalar, value, dtype)

cpdef void from_string(Scalar scalar, str value, Type _):
    scalar._handle = _Scalar(std_string_view_from_py(value))


cpdef void from_binary(Scalar scalar, object value, Type dtype):
    if isinstance(value, np.ndarray):
        value = value.tobytes()
    if len(value) != dtype.size:
        raise ValueError(
            f"Type {dtype} expects a value of size {dtype.size}, "
            f"but the size of value is {len(value)}"
        )
    assert isinstance(value, bytes)
    scalar._handle = _Scalar(dtype._handle, <const char*>(<bytes>value), True)

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
    _Type.Code.BINARY: from_binary,
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

cdef inline tuple to_array_interface(Type ty, tuple shape):
    cdef Type elem_ty = ty.element_type
    shape += (ty.num_elements,)
    if elem_ty.code != _Type.Code.FIXED_ARRAY:
        return (elem_ty.to_numpy_dtype().str, shape)
    return to_array_interface(elem_ty, shape)


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
    _Type.Code.FLOAT16 : lambda Scalar result: half_to_float(
        result._handle.value[__half]()
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
    _Type.Code.STRING : lambda Scalar result: str_from_string_view(
        result._handle.value[std_string_view]()
    ),
    _Type.Code.BINARY: lambda Scalar result: PyMemoryView_FromMemory(
        <char*>(result._handle.ptr()), result._handle.size(), PyBUF_READ
    )
}

cdef class Scalar:
    @staticmethod
    cdef Scalar from_handle(_Scalar handle):
        cdef Scalar result = Scalar.__new__(Scalar)
        result._handle = std_move(handle)
        return result

    def __init__(self, value: Any, dtype: Type | None = None) -> None:
        r"""
        Construct a `Scalar`.

        Parameters
        ----------
        value : Any
            The value to fill the scalar with.
        dtype : Type | None
            The `Type` of the value. If it is `None`, the type will be
            deduced.

        Raises
        ------
        NotImplementedError
            If the type could not be deduced from the object.
        """
        if dtype is None:
            dtype = Type.from_py_object(value)
        try:
            ctor = _CONSTRUCTORS[dtype.code]
        except KeyError:
            raise TypeError(f"unhandled type {dtype}")
        ctor(self, value, dtype)

    cpdef object value(self):
        r"""
        Get the value of the object contained by the `Scalar`.

        Returns
        -------
        Any
            The contained value.

        Raises
        ------
        RuntimeError
            If the contained value could not be reconstructed.
        """
        cdef _Type.Code code

        with nogil:
            code = self._handle.type().code()
        try:
            return _GETTERS[code](self)
        except KeyError as ke:
            raise RuntimeError(f"unhandled type {code}") from ke

    def __str__(self) -> str:
        r"""
        Return a human-readable representation of the scalar.

        Returns
        -------
        str
            The human readable representation of the scalar.
        """
        return f"Scalar({self.value()}, {self.type})"

    def __repr__(self) -> str:
        r"""
        Return a human-readable representation of the scalar.

        Returns
        -------
        str
            The human readable representation of the scalar.
        """
        return str(self)

    @property
    def __array_interface__(self) -> dict[str, Any]:
        r"""
        Retrieve the numpy-compatible array representation of the scalar.

        :returns: The numpy array interface dict.
        :rtype: dict[str, Any]

        :raises ValueError: If the type of the value in the scalar is
        variably sized (e.g. a list).
        """
        cdef Type ty = self.type
        if ty.variable_size:
            raise ValueError(
                "Scalars with variable size types don't support "
                "array interface"
            )
        cdef str dtype_str
        cdef tuple shape
        if ty.code == _Type.Code.FIXED_ARRAY:
            dtype_str, shape = to_array_interface(ty, tuple())
            return {
                "version": 3,
                "shape": shape,
                "typestr": dtype_str,
                "data": (self.ptr, True),
            }
        numpy_type = ty.to_numpy_dtype()
        return {
            "version": 3,
            "shape": (),
            "typestr": numpy_type.str,
            "data": (self.ptr, True),
            "descr": numpy_type.descr,
        }

    @property
    def type(self) -> Type:
        r"""
        Get the type of the scalar.

        :returns: The type of the scalar.
        :rtype: Type
        """
        cdef _Type handle

        with nogil:
            handle = self._handle.type()
        return Type.from_handle(std_move(handle))

    @property
    def ptr(self) -> uintptr_t:
        r"""
        Get the raw pointer to the backing allocation.

        :returns: The pointer to the `Scalar`'s data.
        :rtype: int
        """
        cdef uintptr_t p

        with nogil:
            p = <uintptr_t>self._handle.ptr()
        return p

    @property
    def raw_handle(self) -> uintptr_t:
        r"""
        Get the pointer to the C++ `Scalar` object.

        :returns: The pointer to the C++ `Scalar` object.
        :rtype: int
        """
        return <uintptr_t> &self._handle

    @staticmethod
    def null() -> Scalar:
        r"""
        Construct a nullary `Scalar`.

        A null scalar holds no value, and has the special "null" `Type`. It
        is akin to `None` in Python. It can be passed to tasks, and compared
        against, but its value cannot be obtained.

        Returns
        -------
        Scalar
            A null scalar.
        """
        return Scalar(None, null_type)
