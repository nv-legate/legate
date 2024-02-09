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

from libc.stdint cimport uint32_t
from libcpp.utility cimport move as std_move

from ..utilities.tuple cimport _tuple
from ..utilities.utils cimport is_iterable, uint64_tuple_from_iterable

from collections.abc import Callable, Collection


cdef class Variable:
    @staticmethod
    cdef Variable from_handle(_Variable handle):
        cdef Variable result = Variable.__new__(Variable)
        result._handle = handle
        return result

    def __init__(self) -> None:
        raise ValueError(
            f"{type(self).__name__} objects must not be constructed directly"
        )

    def __str__(self) -> str:
        return self._handle.to_string().decode()

    def __repr__(self) -> str:
        return str(self)


cdef class Constraint:
    @staticmethod
    cdef Constraint from_handle(_Constraint handle):
        cdef Constraint result = Constraint.__new__(Constraint)
        result._handle = handle
        return result

    def __init__(self) -> None:
        raise ValueError(
            f"{type(self).__name__} objects must not be constructed directly"
        )

    def __str__(self) -> str:
        return self._handle.to_string().decode()

    def __repr__(self) -> str:
        return str(self)


cdef class ConstraintProxy:
    r"""A trivial wrapper class to store the function and arguments
    to construct a `Constraint`

    Notes
    -----
    This class is useful to 'defer' construction of the `Constraint` until
    a later time. For example, it is used by `PyTask` to take in Store or Array
    arguments, convert them to the appropriate `Variable`, and then construct
    the `Constraint` transparently.
    """
    def __init__(self, func: Callable[..., Constraint], *args: Any) -> None:
        r"""Construct a `ConstraintProxy`

        Parameters
        ----------
        func : Callable[..., Constraint]
            The function which, given `args`, will construct the `Constraint`.
        *args : Any
            The original arguments to `func`.
        """
        self.func = func
        self.args = args


cpdef object align(VariableOrStr lhs, VariableOrStr rhs):
    if VariableOrStr is Variable:
        return Constraint.from_handle(_align(lhs._handle, rhs._handle))
    # I don't know why cython complains that this is unreachable. It is, just
    # not for every version of this function (and that's the point!!)
    return ConstraintProxy(align, lhs, rhs)


cpdef object broadcast(
    VariableOrStr variable, axes: Collection[int] = tuple()
):
    if not is_iterable(axes):
        raise ValueError("axes must be iterable")

    if VariableOrStr is str:
        return ConstraintProxy(broadcast, variable, axes)

    if len(axes) == 0:
        return Constraint.from_handle(_broadcast(variable._handle))

    cdef _tuple[uint32_t] cpp_axes

    cpp_axes.reserve(len(axes))
    for axis in axes:
        cpp_axes.append_inplace(<uint32_t> axis)
    return Constraint.from_handle(
        _broadcast(variable._handle, std_move(cpp_axes))
    )


cpdef object image(
    VariableOrStr var_function, VariableOrStr var_range
):
    if VariableOrStr is Variable:
        return Constraint.from_handle(
            _image(var_function._handle, var_range._handle)
        )
    return ConstraintProxy(image, var_function, var_range)


cpdef object scale(
    tuple factors,
    VariableOrStr var_smaller,
    VariableOrStr var_bigger
):
    if VariableOrStr is Variable:
        return Constraint.from_handle(
            _scale(
                std_move(uint64_tuple_from_iterable(factors)),
                var_smaller._handle,
                var_bigger._handle,
            )
        )
    return ConstraintProxy(
        scale, factors, var_smaller, var_bigger
    )


cpdef object bloat(
    VariableOrStr var_source,
    VariableOrStr var_bloat,
    tuple low_offsets,
    tuple high_offsets,
):
    if VariableOrStr is Variable:
        return Constraint.from_handle(
            _bloat(
                var_source._handle,
                var_bloat._handle,
                std_move(uint64_tuple_from_iterable(low_offsets)),
                std_move(uint64_tuple_from_iterable(high_offsets)),
            )
        )
    return ConstraintProxy(
        bloat,
        var_source,
        var_bloat,
        low_offsets,
        high_offsets,
    )
