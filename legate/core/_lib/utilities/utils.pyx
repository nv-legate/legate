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

from libcpp.utility cimport move as std_move

from ..utilities.tuple cimport _tuple
from .detail.tuple cimport to_domain_point


cpdef bool is_iterable(object obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


cdef _tuple[uint64_t] uint64_tuple_from_iterable(object obj):
    if not is_iterable(obj):
        raise ValueError(
            f"Expected an iterable but got {type(obj)}"
        )

    cdef _tuple[uint64_t] tpl

    tpl.reserve(len(obj))
    for extent in obj:
        try:
            tpl.append_inplace(<size_t> extent)
        except OverflowError:
            raise ValueError("Extent must be a positive number")

    return std_move(tpl)


cdef _Domain domain_from_iterables(object low, object high):
    return _Domain(
        to_domain_point(uint64_tuple_from_iterable(low)),
        to_domain_point(uint64_tuple_from_iterable(high)),
    )
