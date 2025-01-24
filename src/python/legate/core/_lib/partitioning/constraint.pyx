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

from libc.stdint cimport uint32_t
from libcpp.utility cimport move as std_move

from ..utilities.tuple cimport _tuple
from ..utilities.unconstructable cimport Unconstructable
from ..utilities.utils cimport is_iterable, uint64_tuple_from_iterable

from collections.abc import Callable, Collection


cdef class Variable(Unconstructable):
    @staticmethod
    cdef Variable from_handle(_Variable handle):
        cdef Variable result = Variable.__new__(Variable)
        result._handle = handle
        return result

    def __str__(self) -> str:
        r"""
        Return a human-readable representation of the variable.

        Returns
        -------
        str
            The human readable representation of the variable.
        """
        return self._handle.to_string().decode()

    def __repr__(self) -> str:
        r"""
        Return a human-readable representation of the variable.

        Returns
        -------
        str
            The human readable representation of the variable.
        """
        return str(self)


cdef class Constraint(Unconstructable):
    @staticmethod
    cdef Constraint from_handle(_Constraint handle):
        cdef Constraint result = Constraint.__new__(Constraint)
        result._handle = std_move(handle)
        return result

    def __str__(self) -> str:
        r"""
        Return a human-readable representation of the constraint.

        Returns
        -------
        str
            The human readable representation of the constraint.
        """
        return self._handle.to_string().decode()

    def __repr__(self) -> str:
        r"""
        Return a human-readable representation of the constraint.

        Returns
        -------
        str
            The human readable representation of the constraint.
        """
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
    r"""
    Create an alignment constraint between two variables.

    An alignment constraint between variables `x` and `y` indicates to the
    runtime that the `PhysicalStore`s (leaf-task-local portions, typically
    equal-size tiles) of the `LogicalStore`s corresponding to `x` and `y`
    must have the same global indices (i.e. the stores must "align" with
    one another).

    This is commonly used for e.g. element-wise operations. For example,
    consider an element-wise addition (`z = x + y`), where each array is 100
    elements long. Each leaf task must receive the same local tile for all 3
    arrays. For example, leaf task 0 receives indices `[0, 24)`, leaf task 1
    receives `[25, 49)`, leaf task 2 receives `[50, 74)`, and leaf task 3
    receives `[75, 99)`.

    Parameters
    ----------
    lhs : Variable
        The first variable to align.
    rhs : Variable
        The other variable to align.

    Returns
    -------
    Constraint
        The alignment constraint.
    """
    cdef _Constraint handle

    if VariableOrStr is Variable:
        with nogil:
            handle = _align(lhs._handle, rhs._handle)
        return Constraint.from_handle(std_move(handle))
    # I don't know why cython complains that this is unreachable. It is, just
    # not for every version of this function (and that's the point!!)
    return ConstraintProxy(align, lhs, rhs)


cpdef object broadcast(
    VariableOrStr variable, axes: Collection[int] = tuple()
):
    r"""
    Create a broadcast constraint on a variable.

    A broadcast constraint informs the runtime that the variable should not be
    split among the leaf tasks, instead, each leaf task should get a full copy
    of the underlying store. In other words, the store should be "broadcast"
    in its entirety to all leaf tasks in a task launch.

    In effect, this constraint prevents all dimensions of the store from being
    partitioned.

    Parameters
    ----------
    variable : Variable
        The variable to broadcast.
    axes : Collection[int] (optional)
        An optional set of axes which denotes a subset of the axes of
        `variable` which to broadcast. If given, only the specified axes of
        variable will be broadcast, all other axes will be partitioned
        (subject to any other constraints). If not given (or if empty), all
        axes will be broadcast.

    Returns
    -------
    Constraint
        The broadcast constraint.

    """
    if not is_iterable(axes):
        raise ValueError("axes must be iterable")

    if VariableOrStr is str:
        return ConstraintProxy(broadcast, variable, axes)

    cdef _Constraint handle

    if len(axes) == 0:
        with nogil:
            handle = _broadcast(variable._handle)
        return Constraint.from_handle(std_move(handle))

    cdef _tuple[uint32_t] cpp_axes

    cpp_axes.reserve(len(axes))
    for axis in axes:
        cpp_axes.append_inplace(<uint32_t> axis)

    with nogil:
        handle = _broadcast(variable._handle, std_move(cpp_axes))
    return Constraint.from_handle(std_move(handle))


cpdef object image(
    VariableOrStr var_function,
    VariableOrStr var_range,
    ImageComputationHint hint = ImageComputationHint.NO_HINT,
):
    r"""
    Create an image constraint.

    The elements of `var_function` are treated as pointers to elements in
    `var_range`. Each sub-store `s` of `var_function` is aligned with a
    sub-store `t` of `var_range`, such that every element in `s` will find the
    element of `var_range` it's pointing to inside of `t`.

    An approximate image of a function potentially contains extra points not in
    the function's image. For example, if a function sub-store contains two 2-D
    points `(0, 0)` and `(1, 1)`, the corresponding sub-store of the range
    would only contain the elements at points `(0, 0)` and `(1, 1)` if it was
    constructed from a precise image computation, whereas an approximate image
    computation would yield a sub-store with elements at point `(0, 0)`,
    `(0, 1)`, `(1, 0)`, and `(1, 1)` (two extra elements).

    Currently, the precise image computation can be performed only by CPUs. As
    a result, the function store is copied to the system memory if the store
    was last updated by GPU tasks.  The approximate image computation has no
    such issue and is fully GPU accelerated.

    Parameters
    ----------
    var_function : Variable
        Partition symbol for the function store.
    var_range : Variable
        Partition symbol of the store whose partition should be derived from
        the image.
    hint : ImageComputationHint (optional)
        Hint to the runtime describing how the image computation can be
        performed.  If no hint is given (which is the default), the runtime
        falls back to the precise image computation. Otherwise, the runtime
        computes a potentially approximate image of the function.

    Returns
    -------
    Constraint
        The image constraint.
    """
    cdef _Constraint handle
    if VariableOrStr is Variable:
        with nogil:
            handle = _image(var_function._handle, var_range._handle, hint)
        return Constraint.from_handle(std_move(handle))
    return ConstraintProxy(image, var_function, var_range, hint)


cpdef object scale(
    tuple factors,
    VariableOrStr var_smaller,
    VariableOrStr var_bigger
):
    r"""
    Create a scaling constraint.

    A scaling constraint is similar to an alignment constraint, except that the
    sizes of the aligned tiles is first scaled by `factors`.

    For example, this may be used in compacting a `5x56` array of `bool`s to a
    `5x7` array of bytes, treated as a bitfield. In this case `var_smaller`
    would be the byte array, `var_bigger` would be the array of `bool`s, and
    `factors` would be `[1, 8]` (a `2x3` tile on the byte array corresponds to
    a `2x24` tile on the bool array.

    Formally: if two stores `A` and `B` are constrained by a scaling constraint
    `scale(S, pA, pB)` where `pA` and `pB ` are partition symbols for `A` and
    `B`, respectively, `A` and `B` will be partitioned such that each pair of
    sub-stores `Ak` and `Bk` satisfy the following property:

    ```{math}
    \forall p \in \mathit{dom}(\mathtt{Ak}).
    \forall \delta \in [-\mathtt{L}, \mathtt{H}].

    p + \delta \in \mathit{dom}(\mathtt{Bk})
    \lor p + \delta \not \in \mathit{dom}(\mathtt{B})
    ```

    Parameters
    ----------
    factors : tuple[int, ...]
        The scaling factors.
    var_smaller : Variable
        Partition symbol for the smaller store (i.e., the one whose extents
        are scaled).
    var_bigger : Variable
        Partition symbol for the bigger store.

    Returns
    -------
    Constraint
        The scaling constraint.
    """
    cdef _Constraint handle
    cdef _tuple[uint64_t] tup

    if VariableOrStr is Variable:
        tup = uint64_tuple_from_iterable(factors)
        with nogil:
            handle = _scale(
                std_move(tup), var_smaller._handle, var_bigger._handle,
            )
        return Constraint.from_handle(std_move(handle))
    return ConstraintProxy(
        scale, factors, var_smaller, var_bigger
    )


cpdef object bloat(
    VariableOrStr var_source,
    VariableOrStr var_bloat,
    tuple low_offsets,
    tuple high_offsets,
):
    r"""
    Create a bloat constraint.

    This is typically used in stencil computations, to instruct the runtime
    that the tiles on the "private + ghost" partition (`var_bloat`) must align
    with the tiles on the "private" partition (`var_source`), but also include
    a halo of additional elements off each end.

    For example, if `var_source` and `var_bloat` correspond to 10-element
    vectors, \p var_source is split into 2 tiles, `0-4` and `5-9`,
    `low_offsets == 1` and `high_offsets == 2`, then `var_bloat` will be split
    into 2 tiles, `0-6` and `4-9`.

    Formally, if two stores `A` and `B` are constrained by a bloating
    constraint `bloat(pA, pB, L, H)` where `pA` and `pB ` are partition symbols
    for `A` and `B`, respectively, `A` and `B` will be partitioned such that
    each pair of sub-stores `Ak` and `Bk` satisfy the following property:

    ```{math}
    \forall p \in \mathit{dom}(\mathtt{Ak}).
    \forall \delta \in [-\mathtt{L}, \mathtt{H}].

    p + \delta \in \mathit{dom}(\mathtt{Bk}) \lor
    p + \delta \not \in \mathit{dom}(\mathtt{B})
    ```

    Parameters
    ----------
    var_source : Variable
        Partition symbol for the source store.
    var_bloat : Variable
        Partition symbol for the target store.
    low_offsets : tuple[int, ...]
        Offsets to bloat towards the negative direction.
    high_offsets : tuple[int, ...]
        Offsets to bloat towards the positive direction.

    Returns
    -------
    Constraint
        The bloat constraint.
    """
    cdef _Constraint handle
    cdef _tuple[uint64_t] tup_lo
    cdef _tuple[uint64_t] tup_hi

    if VariableOrStr is Variable:
        tup_lo = uint64_tuple_from_iterable(low_offsets)
        tup_hi = uint64_tuple_from_iterable(high_offsets)

        with nogil:
            handle = _bloat(
                var_source._handle,
                var_bloat._handle,
                std_move(tup_lo),
                std_move(tup_hi),
            )
        return Constraint.from_handle(std_move(handle))
    return ConstraintProxy(
        bloat,
        var_source,
        var_bloat,
        low_offsets,
        high_offsets,
    )
