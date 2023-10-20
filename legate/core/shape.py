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

from __future__ import annotations

from functools import reduce
from typing import Iterable, Iterator, Optional, Union, overload

from typing_extensions import TypeAlias

ExtentLike: TypeAlias = Union["Shape", int, Iterable[int]]


def _cast_tuple(value: int | Iterable[int], ndim: int) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value,) * ndim
    return tuple(value)


class _ShapeComparisonResult(tuple[bool, ...]):
    def __bool__(self) -> bool:
        assert False, "use any() or all()"


class Shape:
    _extents: tuple[int, ...]

    def __init__(
        self,
        extents: Optional[ExtentLike] = None,
    ) -> None:
        """
        Constructs a new shape object

        Parameters
        ----------
        extents: int, Iterable[int], or Shape
           Extents to construct the shape object with
        """
        assert extents is not None
        if isinstance(extents, int):
            self._extents = (extents,)
        else:
            self._extents = tuple(extents)

    @property
    def extents(self) -> tuple[int, ...]:
        """
        Returns the extents of the shape in a tuple

        Returns
        -------
        tuple[int]
            Extents of the shape

        Notes
        -----
        Can block on the producer task
        """
        return self._extents

    def __str__(self) -> str:
        return f"Shape({self._extents})"

    def __repr__(self) -> str:
        return str(self)

    @overload
    def __getitem__(self, idx: int) -> int:
        ...

    @overload
    def __getitem__(self, idx: slice) -> Shape:
        ...

    def __getitem__(self, idx: Union[int, slice]) -> Union[Shape, int]:
        if isinstance(idx, slice):
            return Shape(self.extents[idx])
        else:
            return self.extents[idx]

    def __len__(self) -> int:
        return len(self.extents)

    def __iter__(self) -> Iterator[int]:
        return iter(self.extents)

    def __contains__(self, value: object) -> bool:
        return value in self.extents

    @property
    def fixed(self) -> bool:
        """
        Indicates whether the shape's extents are already computed

        Returns
        ------
        bool
            If ``True``, the shape has fixed extents
        """
        return self._extents is not None

    @property
    def ndim(self) -> int:
        """
        Dimension of the shape. Unlike the ``extents`` property, this is
        non-blocking.

        Returns
        ------
        int
            Dimension of the shape
        """
        return len(self._extents)

    def volume(self) -> int:
        """
        Returns the shape's volume

        Returns
        ------
        int
            Volume of the shape

        Notes
        -----
        Can block on the producer task
        """
        return reduce(lambda x, y: x * y, self.extents, 1)

    def sum(self) -> int:
        """
        Returns a sum of the extents

        Returns
        ------
        int
            Sum of the extents

        Notes
        -----
        Can block on the producer task
        """
        return reduce(lambda x, y: x + y, self.extents, 0)

    def __hash__(self) -> int:
        return hash((self.__class__, True, self._extents))

    def __eq__(self, other: object) -> bool:
        """
        Checks whether the shape is identical to a given shape

        Parameters
        ----------
        other : Shape or Iterable[int]
            Shape to compare with

        Returns
        ------
        bool
            ``True`` if the shapes are identical

        Notes
        -----
        Can block on the producer task
        """
        if isinstance(other, Shape):
            return self.extents == other.extents
        elif isinstance(other, (int, Iterable)):
            lh = self.extents
            rh = (
                other.extents
                if isinstance(other, Shape)
                else _cast_tuple(other, self.ndim)
            )
            return lh == rh
        else:
            return False

    def __le__(self, other: ExtentLike) -> _ShapeComparisonResult:
        """
        Returns the result of element-wise ``<=``.

        Parameters
        ----------
        other : Shape or Iterable[int]
            Shape to compare with

        Returns
        ------
        tuple[bool]
            Result of element-wise ``<=``.

        Notes
        -----
        Can block on the producer task
        """
        lh = self.extents
        rh = (
            other.extents
            if isinstance(other, Shape)
            else _cast_tuple(other, self.ndim)
        )
        assert len(lh) == len(rh)
        return _ShapeComparisonResult(l <= r for (l, r) in zip(lh, rh))

    def __lt__(self, other: ExtentLike) -> _ShapeComparisonResult:
        """
        Returns the result of element-wise ``<``.

        Parameters
        ----------
        other : Shape or Iterable[int]
            Shape to compare with

        Returns
        ------
        tuple[bool]
            Result of element-wise ``<``.

        Notes
        -----
        Can block on the producer task
        """
        lh = self.extents
        rh = (
            other.extents
            if isinstance(other, Shape)
            else _cast_tuple(other, self.ndim)
        )
        assert len(lh) == len(rh)
        return _ShapeComparisonResult(l < r for (l, r) in zip(lh, rh))

    def __ge__(self, other: ExtentLike) -> _ShapeComparisonResult:
        """
        Returns the result of element-wise ``<=``.

        Parameters
        ----------
        other : Shape or Iterable[int]
            Shape to compare with

        Returns
        ------
        tuple[bool]
            Result of element-wise ``<=``.

        Notes
        -----
        Can block on the producer task
        """
        lh = self.extents
        rh = (
            other.extents
            if isinstance(other, Shape)
            else _cast_tuple(other, self.ndim)
        )
        assert len(lh) == len(rh)
        return _ShapeComparisonResult(l >= r for (l, r) in zip(lh, rh))

    def __gt__(self, other: ExtentLike) -> _ShapeComparisonResult:
        """
        Returns the result of element-wise ``<=``.

        Parameters
        ----------
        other : Shape or Iterable[int]
            Shape to compare with

        Returns
        ------
        tuple[bool]
            Result of element-wise ``<=``.

        Notes
        -----
        Can block on the producer task
        """
        lh = self.extents
        rh = (
            other.extents
            if isinstance(other, Shape)
            else _cast_tuple(other, self.ndim)
        )
        assert len(lh) == len(rh)
        return _ShapeComparisonResult(l > r for (l, r) in zip(lh, rh))

    def __add__(self, other: ExtentLike) -> Shape:
        """
        Returns an element-wise addition of the shapes

        Parameters
        ----------
        other : Shape or Iterable[int]
            A shape to add to this shape

        Returns
        ------
        bool
            Result of element-wise addition

        Notes
        -----
        Can block on the producer task
        """
        lh = self.extents
        rh = (
            other.extents
            if isinstance(other, Shape)
            else _cast_tuple(other, self.ndim)
        )
        return Shape(tuple(a + b for (a, b) in zip(lh, rh)))

    def __sub__(self, other: ExtentLike) -> Shape:
        """
        Returns an element-wise subtraction between the shapes

        Parameters
        ----------
        other : Shape or Iterable[int]
            A shape to subtract from this shape

        Returns
        ------
        bool
            Result of element-wise subtraction

        Notes
        -----
        Can block on the producer task
        """
        lh = self.extents
        rh = (
            other.extents
            if isinstance(other, Shape)
            else _cast_tuple(other, self.ndim)
        )
        return Shape(tuple(a - b for (a, b) in zip(lh, rh)))

    def __mul__(self, other: ExtentLike) -> Shape:
        """
        Returns an element-wise multiplication of the shapes

        Parameters
        ----------
        other : Shape or Iterable[int]
            A shape to multiply with this shape

        Returns
        ------
        bool
            Result of element-wise multiplication

        Notes
        -----
        Can block on the producer task
        """
        lh = self.extents
        rh = (
            other.extents
            if isinstance(other, Shape)
            else _cast_tuple(other, self.ndim)
        )
        return Shape(tuple(a * b for (a, b) in zip(lh, rh)))

    def __mod__(self, other: ExtentLike) -> Shape:
        """
        Returns the result of element-wise modulo operation

        Parameters
        ----------
        other : Shape or Iterable[int]
            Shape to compare with

        Returns
        ------
        bool
            Result of element-wise modulo operation

        Notes
        -----
        Can block on the producer task
        """
        lh = self.extents
        rh = (
            other.extents
            if isinstance(other, Shape)
            else _cast_tuple(other, self.ndim)
        )
        return Shape(tuple(a % b for (a, b) in zip(lh, rh)))

    def __floordiv__(self, other: ExtentLike) -> Shape:
        """
        Returns the result of element-wise integer division

        Parameters
        ----------
        other : Shape or Iterable[int]
            A shape to divide this shape by

        Returns
        ------
        bool
            Result of element-wise integer division

        Notes
        -----
        Can block on the producer task
        """
        lh = self.extents
        rh = (
            other.extents
            if isinstance(other, Shape)
            else _cast_tuple(other, self.ndim)
        )
        return Shape(tuple(a // b for (a, b) in zip(lh, rh)))

    def drop(self, dim: int) -> Shape:
        """
        Removes a dimension from the shape

        Parameters
        ----------
        dim : int
            Dimension to remove

        Returns
        ------
        Shape
            Shape with one less dimension

        Notes
        -----
        Can block on the producer task
        """
        extents = self.extents
        return Shape(extents[:dim] + extents[dim + 1 :])

    def update(self, dim: int, new_value: int) -> Shape:
        """
        Replaces the extent of a dimension with a new extent

        Parameters
        ----------
        dim : int
            Dimension to replace

        new_value : int
            New extent

        Returns
        ------
        Shape
            Shape with the chosen dimension updated

        Notes
        -----
        Can block on the producer task
        """
        return self.replace(dim, (new_value,))

    def replace(self, dim: int, new_values: Iterable[int]) -> Shape:
        """
        Replaces a dimension with multiple dimensions

        Parameters
        ----------
        dim : int
            Dimension to replace

        new_values : Iterable[int]
            Extents of the new dimensions

        Returns
        ------
        Shape
            Shape with the chosen dimension replaced

        Notes
        -----
        Can block on the producer task
        """
        if not isinstance(new_values, tuple):
            new_values = tuple(new_values)
        extents = self.extents
        return Shape(extents[:dim] + new_values + extents[dim + 1 :])

    def insert(self, dim: int, new_value: int) -> Shape:
        """
        Inserts a new dimension

        Parameters
        ----------
        dim : int
            Location to insert the new dimension

        new_value : int
            Extent of the new dimension

        Returns
        ------
        Shape
            Shape with one more dimension

        Notes
        -----
        Can block on the producer task
        """
        extents = self.extents
        return Shape(extents[:dim] + (new_value,) + extents[dim:])

    def map(self, mapping: tuple[int, ...]) -> Shape:
        """
        Applies a mapping to each extent in the shape

        Parameters
        ----------
        maping : tuple[int]
            New values for dimensions

        Returns
        ------
        Shape
            Shape with the extents replaced

        Notes
        -----
        Can block on the producer task
        """
        return Shape(tuple(self[mapping[dim]] for dim in range(self.ndim)))

    def strides(self) -> Shape:
        """
        Computes strides of the shape. The last dimension is considered the
        most rapidly changing one. For example, if the shape is ``(3, 4, 5)``,
        the strides are

        ::

            (20, 5, 1)

        Returns
        ------
        Shape
            Strides of the shape

        Notes
        -----
        Can block on the producer task
        """
        strides: tuple[int, ...] = ()
        stride = 1
        for size in reversed(self.extents):
            strides += (stride,)
            stride *= size
        return Shape(reversed(strides))
