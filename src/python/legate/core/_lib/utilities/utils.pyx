# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp.utility cimport move as std_move

from ..utilities.tuple cimport _tuple


cpdef bool is_iterable(object obj):
    r"""
    Determine whether an object is iterable.

    Parameters
    ----------
    obj : Any
        The object to check.

    Returns
    -------
    bool
        `True` if `obj` is iterable, `False` otherwise.
    """
    try:
        iter(obj)
        return True
    except TypeError:
        return False


cdef _tuple[AnyT] tuple_from_iterable(
    object obj, AnyT type_deduction_dummy = 0
):
    if not is_iterable(obj):
        m = f"Expected an iterable but got {type(obj)}"
        raise ValueError(m)

    cdef _tuple[AnyT] tpl

    tpl.reserve(len(obj))
    for extent in obj:
        try:
            tpl.append_inplace(<size_t> extent)
        except OverflowError:
            raise ValueError("Extent must be a positive number")

    return std_move(tpl)


cdef _tuple[uint64_t] uint64_tuple_from_iterable(object obj):
    return tuple_from_iterable[uint64_t](obj)
