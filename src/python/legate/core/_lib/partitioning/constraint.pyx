# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cimport cython

from libc.stdint cimport uint32_t, uint64_t
from libcpp.utility cimport move as std_move
from libcpp.optional cimport optional as std_optional

from ..._ext.task.type cimport ParamList

from ..utilities.tuple cimport _tuple
from ..utilities.unconstructable cimport Unconstructable
from ..utilities.utils cimport is_iterable, tuple_from_iterable
from ..utilities.tuple cimport _tuple

from .proxy cimport (
    _ProxyArrayArgument,
    _ProxyArrayArgumentKind,
    _ProxyConstraint,
    _inputs,
    _outputs,
    _reductions
)

from collections.abc import Collection


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


cdef _ProxyArrayArgument _deduce_arg(
    str arg,
    tuple[tuple[_ProxyArrayArgumentKind, ParamList], ...] task_args
):
    cdef _ProxyArrayArgumentKind kind
    cdef ParamList plist
    cdef uint32_t idx

    for kind, plist in task_args:
        try:
            idx = plist.index(arg)
        except ValueError:
            continue

        if kind == _ProxyArrayArgumentKind.INPUT:
            return _inputs[idx]
        if kind == _ProxyArrayArgumentKind.OUTPUT:
            return _outputs[idx]
        if kind == _ProxyArrayArgumentKind.REDUCTION:
            return _reductions[idx]
        raise ValueError(kind)  # pragma: no cover

    # Actually, we should never get here, because we check if the argument is
    # in param_names before we ever call this function, but good to have this
    # backstop nonetheless.
    m = f"Could not find {arg} in task arguments: {task_args}"
    raise ValueError(m)


cdef class DeferredConstraint:
    r"""A trivial wrapper class to store the function and arguments
    to construct a `Constraint`

    Notes
    -----
    This class is useful to 'defer' construction of the `Constraint` until
    a later time. For example, it is used by `PyTask` to take in Store or Array
    arguments, convert them to the appropriate `Variable`, and then construct
    the `Constraint` transparently.

    While this class is documented here, it is considered an implementation
    detail and is subject to change at any time.
    """
    @staticmethod
    cdef DeferredConstraint construct(
        _ProxyConstraint(*func)(tuple, tuple),
        tuple args
    ):
        cdef DeferredConstraint obj = DeferredConstraint.__new__(
            DeferredConstraint
        )

        obj.func = func
        obj.args = args
        return obj


cdef _ProxyConstraint _handle_align(
    tuple[str, str] args,
    tuple[tuple[_ProxyArrayArgumentKind, ParamList], ...] task_args
):
    if len(args) != 2:
        m = f"align got unexpected arguments '{args}'"  # pragma: no cover
        raise ValueError(m)  # pragma: no cover

    cdef _ProxyArrayArgument lhs_ret, rhs_ret

    lhs_ret = _deduce_arg(args[0], task_args)
    rhs_ret = _deduce_arg(args[1], task_args)
    return _proxy_align(lhs_ret, rhs_ret)


cdef object iter_skip_first(tuple iterable):
    it = iter(iterable)
    next(it)
    return it


cdef list align_handle_variables(Variable first, tuple[Variable, ...] variables):
    cdef list ret = []
    cdef _Constraint handle

    for var in iter_skip_first(variables):
        if not isinstance(var, Variable):
            m = (
                "All variables for alignment must be variables, "
                f"not strings, have {variables}"
            )
            raise TypeError(m)

        with nogil:
            handle = _align(first._handle, (<Variable> var)._handle)

        ret.append(Constraint.from_handle(std_move(handle)))

    return ret

cdef list align_handle_strings(str first, tuple[Variable, ...] variables):
    cdef list ret = []

    for var in iter_skip_first(variables):
        if not isinstance(var, str):
            m = (
                "All variables for alignment must be strings, "
                f"not variables, have {variables}"
            )
            raise TypeError(m)

        ret.append(DeferredConstraint.construct(_handle_align, (first, var)))

    return ret


def align(*variables: Variable | str) -> list[Constraint | DeferredConstraint]:
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
    *variables: Variable
        The set of variables to align.

    Returns
    -------
    list[Constraint]
        The alignment constraint(s).
    """
    if len(variables) < 2:
        return []

    first = variables[0]

    if isinstance(first, str):
        return align_handle_strings(first, variables)

    if isinstance(first, Variable):
        return align_handle_variables(first, variables)

    raise TypeError(type(first))


cdef _ProxyConstraint _handle_bcast(
    tuple[str, tuple[int, ...]] args,
    tuple[tuple[_ProxyArrayArgumentKind, ParamList], ...] task_args
):
    if len(args) != 2:
        m = f"broadcast got unexpected arguments '{args}'"  # pragma: no cover
        raise ValueError(m)  # pragma: no cover

    cdef _ProxyArrayArgument var
    cdef tuple[int, ...] axes
    cdef std_optional[_tuple[uint32_t]] cpp_axes

    var = _deduce_arg(args[0], task_args)
    if axes := args[1]:
        cpp_axes.emplace(tuple_from_iterable[uint32_t](axes))
    return _proxy_broadcast(var, std_move(cpp_axes))


# Cython complains that the DeferredConstraint() path is unreachable if
# VariableOrStr is Variable. That is... obviously the point, so we silence the
# warning here.
@cython.warn.unreachable(False)
def _broadcast_impl(
    VariableOrStr variable, axes: Collection[int] = tuple()
) -> Constraint | DeferredConstraint:
    if not is_iterable(axes):
        raise ValueError("axes must be iterable")

    if VariableOrStr is str:
        return DeferredConstraint.construct(
            func=_handle_bcast, args=(variable, axes)
        )

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


def broadcast(
    VariableOrStr variable,
    *rest: Variable | str | Collection[int] |
    tuple[Variable | str, Collection[int]]
) -> list[Constraint | DeferredConstraint]:
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
    *rest : Variable | Collection[int] | tuple[Variable | Collection[int]]
            (optional)
        Either an axes (denoted by a single collection of integer values), or
        additional variable/axes pairs to broadcast.

        If an axes, this denotes a subset of the axes of `variable` which to
        broadcast. If given, only the specified axes of variable will be
        broadcast, all other axes will be partitioned (subject to any other
        constraints). If not given (or if empty), all axes will be broadcast.

    Returns
    -------
    list[Constraint]
        The broadcast constraint(s).


    Notes
    -----
    This routine may be used to generate multiple broadcast constraints in a
    single call, depending on the calling signature. For example:

    >>> broadcast(x, y, z)
    [Broadcast(x), Broadcast(y), Broadcast(z)]
    >>> broadcast(x, (1, 2, 3))
    [Broadcast(x, (1, 2, 3))]
    >>> broadcast(x, y, (z, (1, 2, 3)), (w, (4, 5, 6)))
    [Broadcast(x), Broadcast(y), Broadcast(z, (1, 2, 3)), Broadcast(w, (4, 5, 6))]
    """
    cdef int num_rest = len(rest)

    # Handle the original signature, broadcast(var, [optional axes])
    if num_rest == 0:
        return [_broadcast_impl(variable)]

    # Need to check that rest[0] isn't a variable or str, because user might be
    # doing broadcast(var_x, var_y), which would fall under the new variadic
    # signature.
    if num_rest == 1 and not isinstance(rest[0], (Variable, str)):
        return [_broadcast_impl(variable, rest[0])]

    # If we are here, then we have a case of the new variadic signature.
    cdef list ret = [_broadcast_impl(variable)]

    for obj in rest:
        if isinstance(obj, (Variable, str)):
            ret.append(_broadcast_impl(obj))
        elif isinstance(obj, (tuple, list)):
            ret.append(_broadcast_impl(*obj))
        else:
            raise TypeError(type(obj))

    return ret


cdef _ProxyConstraint _handle_image(
    tuple[str, str, ImageComputationHint] args,
    tuple[tuple[_ProxyArrayArgumentKind, ParamList], ...] task_args
):
    if len(args) != 3:
        m = f"image got unexpected arguments '{args}'"  # pragma: no cover
        raise ValueError(m)  # pragma: no cover

    cdef _ProxyArrayArgument func_ret, range_ret
    cdef ImageComputationHint hint

    func_ret = _deduce_arg(args[0], task_args)
    range_ret = _deduce_arg(args[1], task_args)
    hint = args[2]
    return _proxy_image(func_ret, range_ret, hint)


# Cython complains that the DeferredConstraint() path is unreachable if
# VariableOrStr is Variable. That is... obviously the point, so we silence the
# warning here.
@cython.warn.unreachable(False)
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

    return DeferredConstraint.construct(
        func=_handle_image, args=(var_function, var_range, hint)
    )


cdef _ProxyConstraint _handle_scale(
    tuple[tuple[int, ...], str, str] args,
    tuple[tuple[_ProxyArrayArgumentKind, ParamList], ...] task_args
):
    if len(args) != 3:
        m = f"scale got unexpected arguments '{args}'"  # pragma: no cover
        raise ValueError(m)  # pragma: no cover

    cdef _tuple[uint64_t] factors
    cdef _ProxyArrayArgument smaller_ret, bigger_ret

    factors = tuple_from_iterable[uint64_t](args[0])
    smaller_ret = _deduce_arg(args[1], task_args)
    bigger_ret = _deduce_arg(args[2], task_args)
    return _proxy_scale(std_move(factors), smaller_ret, bigger_ret)


# Cython complains that the DeferredConstraint() path is unreachable if
# VariableOrStr is Variable. That is... obviously the point, so we silence the
# warning here.
@cython.warn.unreachable(False)
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
        tup = tuple_from_iterable[uint64_t](factors)
        with nogil:
            handle = _scale(
                std_move(tup), var_smaller._handle, var_bigger._handle,
            )
        return Constraint.from_handle(std_move(handle))

    return DeferredConstraint.construct(
        func=_handle_scale, args=(factors, var_smaller, var_bigger)
    )

cdef _ProxyConstraint _handle_bloat(
    tuple[str, str, tuple[int, ...], tuple[int, ...]] args,
    tuple[tuple[_ProxyArrayArgumentKind, ParamList], ...] task_args
):
    if len(args) != 4:
        m = f"bloat got unexpected arguments '{args}'"  # pragma: no cover
        raise ValueError(m)  # pragma: no cover

    cdef _ProxyArrayArgument source_ret, bloat_ret
    cdef _tuple[uint64_t] low_offsets, high_offsets

    source_ret = _deduce_arg(args[0], task_args)
    bloat_ret = _deduce_arg(args[1], task_args)
    low_offsets = tuple_from_iterable[uint64_t](args[2])
    high_offsets = tuple_from_iterable[uint64_t](args[3])

    return _proxy_bloat(
        source_ret, bloat_ret, std_move(low_offsets), std_move(high_offsets)
    )


# Cython complains that the DeferredConstraint() path is unreachable if
# VariableOrStr is Variable. That is... obviously the point, so we silence the
# warning here.
@cython.warn.unreachable(False)
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
        tup_lo = tuple_from_iterable[uint64_t](low_offsets)
        tup_hi = tuple_from_iterable[uint64_t](high_offsets)

        with nogil:
            handle = _bloat(
                var_source._handle,
                var_bloat._handle,
                std_move(tup_lo),
                std_move(tup_hi),
            )
        return Constraint.from_handle(std_move(handle))

    return DeferredConstraint.construct(
        func=_handle_bloat,
        args=(var_source, var_bloat, low_offsets, high_offsets)
    )
