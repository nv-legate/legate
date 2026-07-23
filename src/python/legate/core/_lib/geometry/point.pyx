# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int32_t, int64_t
from libcpp.vector cimport vector as std_vector

from collections.abc import Iterator
from operator import index as operator_index

from ..utilities.utils cimport is_iterable


cdef _Point _point_from_iterable(object obj) except *:
    r"""
    Build a ``legate::Point`` from an arbitrary iterable of integers.
    """
    if isinstance(obj, Point):
        return (<Point> obj)._handle

    cdef std_vector[int64_t] coords

    for coord in obj:
        coords.push_back(<int64_t> operator_index(coord))
    return _point_from_coords(coords)


cdef class Point:
    r"""
    A dimension-erased, multi-dimensional point.

    A ``Point`` behaves like a tuple of integer coordinates (it is iterable,
    indexable, sized, hashable and comparable against tuples), but it also
    exposes the full arithmetic and utility interface of ``legate::Point``.
    """

    @staticmethod
    cdef Point from_handle(_Point handle):
        cdef Point result = Point.__new__(Point)
        result._handle = handle
        return result

    @property
    def dim(self) -> int:
        r"""
        :returns: The dimensionality of the point.
        :rtype: int
        """
        return self._handle.dim()

    @property
    def empty(self) -> bool:
        r"""
        :returns: ``True`` if the point is 0-dimensional, ``False`` otherwise.
        :rtype: bool
        """
        return self._handle.empty()

    @property
    def volume(self) -> int:
        return self._handle.volume()

    def __getitem__(self, object idx) -> int:
        cdef int32_t ndim = <int32_t> self._handle.dim()
        cdef int32_t i

        i = <int32_t> operator_index(idx)
        if i < 0:
            i += ndim
        if i < 0 or i >= ndim:
            raise IndexError("point index out of range")
        return self._handle[i]

    def __len__(self) -> int:
        r"""
        :returns: The dimensionality of the point.
        :rtype: int
        """
        return self._handle.size()

    def __iter__(self) -> Iterator[int]:
        r"""
        :returns: An iterator over the point's coordinates.
        :rtype: Iterator[int]
        """
        cdef int32_t ndim = <int32_t> self._handle.dim()
        cdef int32_t i

        for i in range(ndim):
            yield self._handle[i]

    def __eq__(self, object other) -> bool:
        if isinstance(other, Point):
            return self._handle == (<Point> other)._handle
        if is_iterable(other):
            return tuple(self) == tuple(other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(tuple(self))

    def __repr__(self) -> str:
        r"""
        :returns: A human-readable representation of the point.
        :rtype: str
        """
        return f"Point({', '.join(str(coord) for coord in self)})"

    def __str__(self) -> str:
        return repr(self)
