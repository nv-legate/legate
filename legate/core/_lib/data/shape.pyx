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

from libc.stdint cimport int64_t, uint32_t, uint64_t
from libcpp.utility cimport move as std_move

from collections.abc import Iterator

from ..utilities.utils cimport uint64_tuple_from_iterable


cdef class Shape:
    @staticmethod
    cdef Shape from_handle(_Shape handle):
        cdef Shape result = Shape.__new__(Shape)
        result._handle = handle
        result._extents = None
        return result

    @staticmethod
    cdef _Shape from_shape_like(object obj):
        if isinstance(obj, Shape):
            return (<Shape> obj)._handle
        return _Shape(std_move(uint64_tuple_from_iterable(obj)))

    def __init__(self, object obj) -> None:
        self._handle = Shape.from_shape_like(obj)

    @property
    def extents(self) -> tuple[uint64_t, ...]:
        """
        Returns the shape's extents

        Returns
        -------
        tuple[int, ...]
            Dimension extents

        Notes
        -----
        If the shape is of an unbound array or store, the call blocks the
        execution until the shape becomes ready.
        """
        if self._extents is None:
            self._extents = tuple(self._handle.extents().data())
        return self._extents

    @property
    def volume(self) -> uint64_t:
        """
        Returns the shape's volume

        Returns
        -------
        int
            Volume of the shape

        Notes
        -----
        If the shape is of an unbound array or store, the call blocks the
        execution until the shape becomes ready.
        """
        return self._handle.volume()

    @property
    def ndim(self) -> uint32_t:
        """
        Returns the number of dimensions of this shape

        Returns
        -------
        int
            Number of dimensions

        Notes
        -----
        Unlike other shape-related queries, this call is non-blocking
        """
        return self._handle.ndim()

    def __getitem__(self, int64_t idx) -> uint64_t:
        """
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
        return self.extents[idx]

    def __eq__(self, object other) -> bool:
        cdef _Shape other_shape
        try:
            other_shape = Shape.from_shape_like(other)
        except ValueError:
            return NotImplemented
        return self._handle == other_shape

    def __len__(self) -> uint64_t:
        """
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
        return iter(self.extents)

    def __str__(self) -> str:
        return self._handle.to_string().decode()

    def __repr__(self) -> str:
        return str(self)
