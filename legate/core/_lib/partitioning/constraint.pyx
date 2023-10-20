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

from libc.stdint cimport int32_t
from libcpp.utility cimport move as std_move

from ..data.shape cimport _Shape
from ..utilities.tuple cimport tuple as _tuple
from ...utils import is_iterable


cdef class Variable:
    @staticmethod
    cdef Variable from_handle(_Variable handle):
        cdef Variable result = Variable.__new__(Variable)
        result._handle = handle
        return result

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

    def __str__(self) -> str:
        return self._handle.to_string().decode()

    def __repr__(self) -> str:
        return str(self)


def align(Variable lhs, Variable rhs) -> Constraint:
    return Constraint.from_handle(_align(lhs._handle, rhs._handle))


def broadcast(Variable variable, axes = None) -> Constraint:
    if axes is None:
        return Constraint.from_handle(_broadcast(variable._handle))

    if not is_iterable(axes):
        raise ValueError("axes must bean iterable")

    if len(axes) == 0:
        return Constraint.from_handle(_broadcast(variable._handle))

    cdef _tuple[int32_t] cpp_axes
    for axis in axes:
        cpp_axes.append_inplace(<int32_t> axis)
    return Constraint.from_handle(
        _broadcast(variable._handle, std_move(cpp_axes))
    )


def image(Variable var_function, Variable var_range) -> Constraint:
    return Constraint.from_handle(
        _image(var_function._handle, var_range._handle)
    )


def scale(
    tuple factors, Variable var_smaller, Variable var_bigger
) -> Constraint:
    return Constraint.from_handle(
        _scale(
            _Shape(factors),
            var_smaller._handle,
            var_bigger._handle,
        )
    )


def bloat(
    Variable var_source,
    Variable var_bloat,
    tuple low_offsets,
    tuple high_offsets,
) -> Constraint:
    return Constraint.from_handle(
        _bloat(
            var_source._handle,
            var_bloat._handle,
            _Shape(low_offsets),
            _Shape(high_offsets),
        )
    )
