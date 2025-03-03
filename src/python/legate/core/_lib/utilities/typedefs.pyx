# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from libc.stdint cimport int32_t

from typing import NewType
from operator import index as operator_index

LocalTaskID = NewType("LocalTaskID", int)
GlobalTaskID = NewType("GlobalTaskID", int)

LocalRedopID = NewType("LocalRedopID", int)
GlobalRedopID = NewType("GlobalRedopID", int)

cdef class DomainPoint:
    @staticmethod
    cdef DomainPoint from_handle(_DomainPoint handle):
        cdef DomainPoint result = DomainPoint.__new__(DomainPoint)
        result._handle = handle
        return result

    def __init__(self):
        r"""
        Construct an empty `DomainPoint`.
        """
        self._handle = _DomainPoint()

    @property
    def dim(self) -> int32_t:
        r"""
        Get the number of dimensions of the domain point.

        :returns: The dimension of the point.
        :rtype: int
        """
        return self._handle.get_dim()

    def __getitem__(self, idx: int) -> int:
        r"""
        Get a value in the domain point.

        Parameters
        ----------
        idx : int
            The index to get from.

        Returns
        -------
        int
            The value.
        """
        return self._handle[operator_index(idx)]

    def __setitem__(self, idx: int, coord: int) -> None:
        r"""
        Set a value in the domain point.

        Parameters
        ----------
        idx : int
            The index to set at.
        coord : int
            The value to set.
        """
        self._handle[operator_index(idx)] = coord

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DomainPoint):
            return NotImplemented

        return self._handle == (<DomainPoint> other)._handle

    def __str__(self) -> str:
        r"""
        Get a human-readable string representation of the domain point.

        Returns
        -------
        str
            The string representation.
        """
        cdef int i
        cdef str tmp = ",".join(str(self[i]) for i in range(self.dim))
        return f"<{tmp}>"


cdef class Domain:
    @staticmethod
    cdef Domain from_handle(_Domain handle):
        cdef Domain result = Domain.__new__(Domain)
        result._handle = handle
        return result

    def __init__(self):
        r"""
        Construct an empty `Domain`.
        """
        self._handle = _Domain()

    @property
    def dim(self) -> int32_t:
        r"""
        Get the number of dimensions of the domain.

        :returns: The dimension of the domain.
        :rtype: int
        """
        return self._handle.get_dim()

    @property
    def lo(self) -> DomainPoint:
        r"""
        Get the smallest point in the domain.

        :returns: The point.
        :rtype: DomainPoint
        """
        return DomainPoint.from_handle(self._handle.lo())

    @property
    def hi(self) -> DomainPoint:
        r"""
        Get the largest point in the domain.

        :returns: The point.
        :rtype: DomainPoint
        """
        return DomainPoint.from_handle(self._handle.hi())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Domain):
            return NotImplemented

        return self._handle == (<Domain> other)._handle

    def __str__(self) -> str:
        r"""
        Get a human-readable string representation of the domain point.

        Returns
        -------
        str
            The string representation.
        """
        return f"[{self.lo} ... {self.hi}]"
