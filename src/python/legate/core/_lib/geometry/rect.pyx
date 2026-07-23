# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int32_t
from libcpp cimport bool

from collections.abc import Iterator
from operator import index as operator_index

from .point cimport Point, _point_from_iterable


cdef _Rect _rect_from_iterables(object lo, object hi) except *:
    r"""
    Build a ``legate::Rect`` from a pair of iterables of integers.
    """
    return _Rect(_point_from_iterable(lo), _point_from_iterable(hi))


cdef class Rect:
    r"""
    A dimension-erased, multi-dimensional rectangle (dense domain).

    For backwards compatibility a ``Rect`` behaves like a ``(lo, hi)`` pair: it
    is iterable, indexable and sized as such (``lo, hi = rect`` works), and it
    compares equal to the corresponding tuple of tuples. In addition, it
    exposes the full interface of ``legate::Rect``. To iterate over the points
    contained in the domain, use :meth:`iterate` instead.
    """

    @staticmethod
    cdef Rect from_handle(_Rect handle):
        cdef Rect result = Rect.__new__(Rect)
        result._handle = handle
        return result

    @property
    def lo(self) -> Point:
        r"""
        :returns: The lower bound of the domain.
        :rtype: Point
        """
        return Point.from_handle(self._handle.lo())

    @property
    def hi(self) -> Point:
        r"""
        :returns: The upper bound of the domain.
        :rtype: Point
        """
        return Point.from_handle(self._handle.hi())

    @property
    def dim(self) -> int:
        r"""
        :returns: The dimensionality of the domain.
        :rtype: int
        """
        return self._handle.dim()

    @property
    def empty(self) -> bool:
        r"""
        :returns: ``True`` if the domain contains no points.
        :rtype: bool
        """
        return self._handle.empty()

    @property
    def volume(self) -> int:
        r"""
        :returns: The number of points in the domain.
        :rtype: int
        """
        return self._handle.volume()

    @property
    def extents(self) -> Point:
        r"""
        :returns: A point whose coordinates are the extents (``hi - lo + 1``)
                  of each dimension.
        :rtype: Point
        """
        return Point.from_handle(self._handle.extents())

    @property
    def strides(self) -> Point:
        r"""
        :returns: A point whose coordinates are the C-order strides of each
                  dimension.
        :rtype: Point
        """
        return Point.from_handle(self._handle.strides())

    def __getitem__(self, object idx) -> Point:
        r"""
        Return ``lo`` for index ``0`` and ``hi`` for index ``1``.
        """
        if isinstance(idx, slice):
            return (self.lo, self.hi)[idx]

        cdef int32_t i = <int32_t> operator_index(idx)

        if i == 0 or i == -2:
            return self.lo
        if i == 1 or i == -1:
            return self.hi
        raise IndexError("Rect index out of range")

    def __iter__(self) -> Iterator[Point]:
        r"""
        :returns: An iterator yielding the ``lo`` and ``hi`` bounds.
        :rtype: Iterator[Point]
        """
        yield self.lo
        yield self.hi

    def __len__(self) -> int:
        r"""
        :returns: Always ``2`` (the ``lo`` and ``hi`` bounds).
        :rtype: int
        """
        return 2

    def __eq__(self, object other) -> bool:
        if isinstance(other, Rect):
            return self._handle == (<Rect> other)._handle

        try:
            pair = tuple(other)
        except TypeError:
            return NotImplemented
        if len(pair) != 2:
            return NotImplemented
        return self.lo == pair[0] and self.hi == pair[1]

    def __hash__(self) -> int:
        return hash((tuple(self.lo), tuple(self.hi)))

    def __repr__(self) -> str:
        r"""
        :returns: A human-readable representation of the domain.
        :rtype: str
        """
        return f"Rect(lo={tuple(self.lo)!r}, hi={tuple(self.hi)!r})"

    def __str__(self) -> str:
        return repr(self)
