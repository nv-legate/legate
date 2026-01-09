# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint64_t
from libcpp.vector cimport vector as std_vector

from typing import NewType, TypeAlias
from collections import namedtuple

from .detail.tuple cimport _to_domain_point
from .utils cimport tuple_from_iterable
from .tuple cimport _tuple

LocalTaskID = NewType("LocalTaskID", int)
GlobalTaskID = NewType("GlobalTaskID", int)

LocalRedopID = NewType("LocalRedopID", int)
GlobalRedopID = NewType("GlobalRedopID", int)

DomainPoint: TypeAlias = tuple[int, ...]
Domain = namedtuple("Domain", ("lo", "hi"), defaults=((0,), (0,)))

cdef _DomainPoint domain_point_from_iterable(object iterable):
    cdef _tuple[uint64_t] tup = tuple_from_iterable[uint64_t](iterable)

    return _to_domain_point(tup)

cdef DomainPoint_t domain_point_to_py(const _DomainPoint& point):
    cdef int dim = point.get_dim()
    cdef int i
    cdef std_vector[uint64_t] vec

    vec.reserve(dim)
    for i in range(dim):
        vec.push_back(point[i])

    return tuple(vec)

cdef _Domain domain_from_iterables(object low, object high):
    cdef _DomainPoint lo = domain_point_from_iterable(low)
    cdef _DomainPoint hi = domain_point_from_iterable(high)

    return _Domain(lo, hi)


cdef Domain_t domain_to_py(const _Domain& domain):
    cdef DomainPoint_t lo = domain_point_to_py(domain.lo())
    cdef DomainPoint_t hi = domain_point_to_py(domain.hi())

    return Domain(lo=lo, hi=hi)
