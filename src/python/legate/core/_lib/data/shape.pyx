# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint32_t, uint64_t
from libcpp.utility cimport move as std_move
from libcpp.vector cimport vector as std_vector

from collections.abc import Collection, Iterator
from operator import index as operator_index

from ..utilities.utils cimport uint64_tuple_from_iterable

cdef class Shape:
    @staticmethod
    cdef Shape from_handle(_Shape handle):
        cdef Shape result = Shape.__new__(Shape)
        result._handle = std_move(handle)
        result._extents = None
        return result

    @staticmethod
    cdef _Shape from_shape_like(object obj):
        if isinstance(obj, Shape):
            return (<Shape> obj)._handle
        return _Shape(std_move(uint64_tuple_from_iterable(obj)))

    def __init__(self, obj: Shape | Collection[int]) -> None:
        r"""
        Construct a `Shape`.

        Parameters
        ----------
        obj : Shape | Collection[int]
            An iterable from which to construct the new shape from.
        """
        self._handle = Shape.from_shape_like(obj)

    @property
    def extents(self) -> tuple[uint64_t, ...]:
        r"""
        Returns the shape's extents

        :returns: Dimension extents
        :rtype: tuple[int, ...]

        Notes
        -----
        If the shape is of an unbound array or store, the call blocks the
        execution until the shape becomes ready.
        """
        cdef const std_vector[uint64_t]* v = NULL

        if self._extents is None:
            with nogil:
                v = &self._handle.extents().data()
            self._extents = tuple(v[0])
        return self._extents

    @property
    def volume(self) -> uint64_t:
        r"""
        Returns the shape's volume

        :returns: Volume of the shape
        :rtype: int

        Notes
        -----
        If the shape is of an unbound array or store, the call blocks the
        execution until the shape becomes ready.
        """
        cdef uint64_t ret = 0

        with nogil:
            ret = self._handle.volume()
        return ret

    @property
    def ndim(self) -> uint32_t:
        r"""
        Returns the number of dimensions of this shape

        :returns: Number of dimensions
        :rtype: int

        Notes
        -----
        Unlike other shape-related queries, this call is non-blocking
        """
        cdef uint32_t ret = 0

        with nogil:
            ret = self._handle.ndim()
        return ret

    def __getitem__(self, idx: int) -> int:
        r"""
        Returns the extent of a given dimension

        Parameters
        ----------
        idx
            Dimension index

        Returns
        -------
        int
            Extent of the chosen dimension

        Raises
        ------
        IndexError
            If the dimension index is out-of-range

        Notes
        -----
        If the shape is of an unbound array or store, the call blocks the
        execution until the shape becomes ready.
        """
        return self.extents[operator_index(idx)]

    def __eq__(self, object other) -> bool:
        r"""
        Return whether two shapes are equal.

        Parameters
        ----------
        other : Any
            The rhs to compare against.

        Returns
        -------
        bool
            `True` if this shape is equal to `other`, `False` otherwise.
        """
        cdef _Shape other_shape
        try:
            other_shape = Shape.from_shape_like(other)
        except ValueError:
            return NotImplemented

        cdef bool ret

        with nogil:
            ret = self._handle == other_shape
        return ret

    def __len__(self) -> uint64_t:
        r"""
        Returns the number of dimensions of this shape

        Returns
        -------
        int
            Number of dimensions

        Notes
        -----
        Unlike other shape-related queries, this call is non-blocking
        """
        return self.ndim

    def __iter__(self) -> Iterator[int]:
        r"""
        Retrun an iterator to the shapes extents.

        Returns
        -------
        Iterator[int]
            An iterator to the shapes extents.
        """
        return iter(self.extents)

    def __str__(self) -> str:
        r"""
        Return a human-readable representation of the shape.

        Returns
        -------
        str
            The human readable representation of the shape.
        """
        return self._handle.to_string().decode()

    def __repr__(self) -> str:
        r"""
        Return a human-readable representation of the shape.

        Returns
        -------
        str
            The human readable representation of the shape.
        """
        return str(self)
