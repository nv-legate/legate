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
from typing import Any

import numpy as np
from numpy.typing import DTypeLike


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
    _Type.Code.NIL : np.dtype(object),
}

cdef dict _TO_SIZED_NUMPY_DTYPES = {
    _Type.Code.STRING : lambda int size: np.dtype((np.str_, size)),
    _Type.Code.BINARY : lambda int size: np.dtype((np.bytes_, size)),
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
    # None-arrays are all np.object but np.object could be literally *anything*
    # (including ragged arrays), so the mapping is not bijective. Hence we
    # cannot do the "backwards" mapping.
    # np.dtype(object) : null_type
}

cdef dict _FROM_SIZED_NUMPY_DTYPES = {
    np.dtype(np.str_) : lambda int size: string_type,
    np.dtype(np.bytes_) : lambda int size: binary_type(size)
}

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
    return Type.from_np_dtype(np.min_scalar_type(py_object))

cdef inline Type deduce_array_type_(object py_object):
    cdef uint32_t n_sub = 0

    try:
        n_sub = len(py_object)
    except TypeError:
        # If we have a numpy array, then we may be also dealing with a scalar
        # numpy array. Scalar numpy arrays of *unsized* objects (like bytes,
        # void pointers etc.) will error with "TypeError len() of unsized
        # object" if you try to take their length. We will handle that
        # correctly later, so it suffices to just take the length of the
        # top-level shape.
        if py_object.ndim == 0 and py_object.size > 0:
            # We have a numpy scalar
            py_object = py_object.reshape(py_object.size)
            n_sub = len(py_object)

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
        result._handle = std_move(ty)
        return result

    def __init__(self) -> None:
        r"""
        Construct a `Type` (by default a "null" type).
        """
        self._handle = _null_type()

    @property
    def code(self) -> TypeCode:
        r"""
        Get the type code of the type.

        :returns: The type code.
        :rtype: TypeCode
        """
        cdef TypeCode ret

        with nogil:
            ret = self._handle.code()
        return ret

    @property
    def size(self) -> uint32_t:
        r"""
        Get the size (in bytes) of the data type.

        :returns: The size of the data type.
        :rtype: int
        """
        cdef uint32_t ret

        with nogil:
            ret = self._handle.size()
        return ret

    @property
    def alignment(self) -> uint32_t:
        r"""
        Get the alignmenent (in bytes) of the data type.

        :returns: The alignment of the data type.
        :rtype: int
        """
        cdef uint32_t ret

        with nogil:
            ret = self._handle.alignment()
        return ret

    @property
    def uid(self) -> int32_t:
        r"""
        Get the UID (universal ID) of the type.

        All types which share a UID are guaranteed to be equal.

        :returns: The UID of the type.
        :rtype: int
        """
        cdef int32_t ret

        with nogil:
            ret = self._handle.uid()
        return ret

    @property
    def variable_size(self) -> bool:
        r"""
        Get whether the data type is variably sized.

        :returns: `True` if the type is variable size, `False` otherwise.
        :rtype: bool
        """
        cdef bool ret

        with nogil:
            ret = self._handle.variable_size()
        return ret

    @property
    def is_primitive(self) -> bool:
        r"""
        Get whether a type is "primitive".

        Primitive types are generally those that are "built in", such as
        `int32_t`, `uin64_t`, `float`, `double`, `int`, etc. Struct types, list
        types, and other such compound types are not primitive.

        :returns: `True` if the type is primitive, `False` otherwise.
        :rtype: bool
        """
        cdef bool ret

        with nogil:
            ret = self._handle.is_primitive()
        return ret

    cpdef void record_reduction_op(
        self, int32_t op_kind, _GlobalRedopID reduction_op_id
    ):
        r"""
        Record a reduction operator.

        The global ID of the reduction operator is issued when that operator is
        registered to the runtime.

        Parameters
        ----------
        op_kind : ReductionOpKind | int
            The reduction operator kind.
        reduction_op_id : GlobalRedopID
            The global reduction ID.
        """
        with nogil:
            self._handle.record_reduction_operator(op_kind, reduction_op_id)

    cpdef _GlobalRedopID reduction_op_id(self, int32_t op_kind):
        r"""
        Finds the global redop ID for a given reduction op kind.

        Parameters
        ----------
        op_kind : ReductionOpKind | int
            The redop kind.

        Returns
        -------
        GlobalRedopID
            The global redop ID.

        Raises
        ------
        ValueError
            If `op_kind` does not exist for this type.
        """
        with nogil:
            return self._handle.find_reduction_operator(op_kind)

    def __repr__(self) -> str:
        r"""
        Return a human-readable string representation of the type.

        Returns
        -------
        str
            The string representation.
        """
        return self._handle.to_string().decode()

    cpdef object to_numpy_dtype(self):
        r"""
        Get the equivalent numpy `dtype` for this type.

        Returns
        -------
        np.dtype
            The numpy dtype.

        Raises
        ------
        ValueError
            If the equivalent numpy `dtype` could not be determined.
        """
        cdef _Type.Code code = self.code

        try:
            return _TO_NUMPY_DTYPES[code]
        except KeyError:
            pass

        try:
            return _TO_SIZED_NUMPY_DTYPES[code](self.size)
        except KeyError:
            pass

        raise ValueError(f"Invalid type code: {code}")

    @property
    def raw_ptr(self) -> uintptr_t:
        r"""
        Get a raw pointer to the C++ `Type` object.

        :returns: The pointer to the C++ object.
        :rtype: int
        """
        return <uintptr_t>(&self._handle)

    def __hash__(self) -> int:
        r"""
        Compute the hash of the type.

        :returns: The hash.
        :rtype: int
        """
        return self.uid

    def __eq__(self, object other) -> bool:
        r"""
        Compate this type against another.

        Parameters
        ----------
        other : Any
            The other to compare against.

        Returns
        -------
        bool
            `True` if this type equals `other`, `False` otherwise.
        """
        if isinstance(other, Type):
            return self._handle == (<Type> other)._handle
        return NotImplemented

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
            return Type.from_np_dtype(py_object.dtype)
        if isinstance(py_object, (bytes, memoryview)):
            return binary_type(len(py_object))
        if py_object is None:
            return null_type
        raise NotImplementedError(f"unsupported type: {py_object!r}")

    # Have to do it this way because @staticmethod is not yet supported on
    # cpdef functions. And I don't want to lose type information for cdef
    # functions by making from_py_object() into a full-blown def.
    @staticmethod
    def from_python_object(py_object: Any) -> Type:
        r"""Construct a ``Type`` object by deducing the type of a python
        object.

        Parameters
        ----------
        py_object : Any
            The python object to type-deduce.

        Returns
        -------
        Type
            The deduced type.

        Raises
        ------
        NotImplementedError
            If the type deduction fails.
        """
        return Type.from_py_object(py_object)

    @staticmethod
    cdef Type from_np_dtype(object np_dtype):
        try:
            return _FROM_NUMPY_DTYPES[np_dtype]
        except KeyError:
            pass

        cdef int obj_size = int(
            <str>(np_dtype.str).lstrip(<str>(np_dtype.char) + "<=>|")
        )
        base_dtype = np.dtype(np_dtype.base.type)

        try:
            return _FROM_SIZED_NUMPY_DTYPES[base_dtype](obj_size)
        except KeyError:
            # Either lookup or np_dtype.base failed
            pass

        raise NotImplementedError(f"Unhandled numpy data type: {np_dtype}")

    @staticmethod
    def from_numpy_dtype(dtype: DTypeLike) -> Type:
        r"""Construct a ``Type`` object from a numpy ``dtype``.

        Parameters
        ----------
        dtype : DTypeLike
            The numpy dtype.

        Returns
        -------
        Type
            The deduced type.

        Raises
        ------
        NotImplementedError
            If type deduction fails.
        """
        return Type.from_np_dtype(dtype)

cdef class FixedArrayType(Type):
    # Cannot use Unconstructable here because Cython only allows 1 extension
    # type base-class
    def __init__(self) -> None:
        raise ValueError(
            f"{type(self).__name__} objects must not be constructed directly"
        )

    @property
    def num_elements(self) -> uint32_t:
        r"""
        Get the number of elements of this array.

        Only gets the "top-level" number of elements. For example, if this type
        refers to an array of arrays:
        ```
        FixedArrayType[
          FixedArrayType[uint32_t, uint32_t, uint32_t, uint32_t],
          FixedArrayType[uint32_t, uint32_t, uint32_t, uint32_t]
        ]
        ```
        this function would return `2`, not `4`.

        :returns: The number of elements.
        :rtype: int
        """
        cdef uint32_t ret

        with nogil:
            ret = self._handle.as_fixed_array_type().num_elements()
        return ret

    @property
    def element_type(self) -> Type:
        r"""
        Get the element type of the array type.

        :returns: The element type.
        :rtype: Type
        """
        cdef _Type handle

        with nogil:
            handle = self._handle.as_fixed_array_type().element_type()
        return Type.from_handle(std_move(handle))

    cpdef object to_numpy_dtype(self):
        r"""
        Get the equivalent numpy `dtype` for this type.

        Returns
        -------
        np.dtype
            The numpy dtype.

        Raises
        ------
        ValueError
            If the equivalent numpy `dtype` could not be determined.
        """
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
        r"""
        Get the number of fields in the struct type.

        Only gets the "top-level" number of fields. For example, if this type
        refers to a struct of structs:
        ```
        StructType[
          StructType[
            uint32_t, float32, complex128, string_type
          ]
          StructType[
            int16_t
          ]
        ]
        ```
        this function would return `2`, not `4` or `1`.

        :returns: The number of fields in the struct type.
        :rtype: int
        """
        cdef uint32_t ret

        with nogil:
            ret = self._handle.as_struct_type().num_fields()
        return ret

    cpdef Type field_type(self, uint32_t field_idx):
        r"""
        Get the type of a particular field.

        Parameters
        ----------
        field_idx : int
            The index of the field to get.

        Returns
        -------
        Type
            The field type.
        """
        cdef _Type handle

        with nogil:
            handle = self._handle.as_struct_type().field_type(field_idx)
        return Type.from_handle(std_move(handle))

    @property
    def aligned(self) -> bool:
        r"""
        Get whether the fields are aligned in memory.

        :returns: `True` if the fields are aligned, `False` otherwise.
        :rtype: bool
        """
        cdef bool ret

        with nogil:
            ret = self._handle.as_struct_type().aligned()
        return ret

    @property
    def offsets(self) -> tuple[uint32_t, ...]:
        r"""
        Get the memory offsets (in bytes) for each field in the struct.

        :returns: The offsets of the fields.
        :rtype: tuple[int, ...]
        """
        cdef std_vector[uint32_t] v

        with nogil:
            v = self._handle.as_struct_type().offsets()
        return tuple(v)

    cpdef object to_numpy_dtype(self):
        r"""
        Get the equivalent numpy `dtype` for this type.

        Returns
        -------
        np.dtype
            The numpy dtype.

        Raises
        ------
        ValueError
            If the equivalent numpy `dtype` could not be determined.
        """
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
    r"""
    Create a binary type.

    Parameters
    ----------
    size : int
        The size (in bytes) of the type.

    Returns
    -------
    Type
        The binary type.
    """
    return Type.from_handle(_binary_type(size))


cpdef FixedArrayType array_type(Type element_type, uint32_t N):
    r"""
    Create a fixed array type.

    Parameters
    ----------
    element_type : Type
        The type of the elements.
    N : int
        The number of elements.

    Returns
    -------
    FixedArrayType
        The fixed array type.
    """
    return <FixedArrayType> Type.from_handle(
        _fixed_array_type(element_type._handle, N)
    )


cpdef StructType struct_type(list field_types, bool align = False):
    r"""
    Create a structure type.

    Parameters
    ----------
    field_types : list[Type]
        The types of the fields.
    align : bool (`False`)
        Whether the types should be aligned in memory.

    Returns
    -------
    StructType
        The structure type.
    """
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
    r"""
    Create a point type.

    Parameters
    ----------
    dim : int
        The number of dimensions of the point.

    Returns
    -------
    FixedArrayType
        The point type.
    """
    return FixedArrayType.from_handle(_point_type(ndim))


cpdef StructType rect_type(int32_t ndim):
    r"""
    Create a rect type.

    Parameters
    ----------
    ndim : int
        The number of dimensions of rect.

    Returns
    -------
    StructType
        The rect type.
    """
    return StructType.from_handle(_rect_type(ndim))
