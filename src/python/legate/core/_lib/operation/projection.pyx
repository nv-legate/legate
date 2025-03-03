# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int32_t
from libcpp.utility cimport move as std_move

from ..utilities.unconstructable cimport Unconstructable


cdef class SymbolicExpr(Unconstructable):
    @staticmethod
    cdef SymbolicExpr from_handle(_SymbolicExpr handle):
        cdef SymbolicExpr result = SymbolicExpr.__new__(SymbolicExpr)
        result._handle = std_move(handle)
        return result

    @property
    def dim(self) -> int:
        r"""
        Get the dimension index of this expression.

        :returns: The dimension.
        :rtype: int
        """
        cdef int ret

        with nogil:
            ret = self._handle.dim()
        return ret

    @property
    def weight(self) -> int:
        r"""
        Get the weight for the coordinates.

        :returns: The weight.
        :rtype: int
        """
        cdef int ret

        with nogil:
            ret = self._handle.weight()
        return ret

    @property
    def offset(self) -> int:
        r"""
        Get the offset of the expression.

        :returns: The offset.
        :rtype: int
        """
        cdef int ret

        with nogil:
            ret = self._handle.offset()
        return ret

    cpdef bool is_identity(self, int32_t dim):
        r"""
        For a given dimension, return whether the expression is an identity
        mapping.

        Parameters
        ----------
        dim : int
            The dimension to check

        Returns
        -------
        bool
            `True` if the expression is the identity, `False` otherwise.
        """
        cdef bool ret

        with nogil:
            ret = self._handle.is_identity(dim)
        return ret

    def __eq__(self, object other) -> bool:
        if isinstance(other, SymbolicExpr):
            return self._handle == (<SymbolicExpr> other)._handle
        return NotImplemented

    def __mul__(self, int32_t other) -> SymbolicExpr:
        r"""
        Multiply an expression by a scaling factor.

        Parameters
        ----------
        other : int
            The scaling factor.

        Returns
        -------
        SymbolicExpr
            The result of the multiplication.
        """
        cdef _SymbolicExpr handle

        with nogil:
            handle = self._handle * other
        return SymbolicExpr.from_handle(std_move(handle))

    def __add__(self, int32_t other) -> SymbolicExpr:
        r"""
        Shift an expression by a scaling factor.

        Parameters
        ----------
        other : int
            The scaling factor.

        Returns
        -------
        SymbolicExpr
            The result of the shift.
        """
        cdef _SymbolicExpr handle

        with nogil:
            handle = self._handle + other
        return SymbolicExpr.from_handle(std_move(handle))

    def __str__(self) -> str:
        r"""
        Return a human-readable representation of the expression.

        Returns
        -------
        str
            The human readable representation of the expression.
        """
        return self._handle.to_string().decode()

    def __repr__(self) -> str:
        r"""
        Return a human-readable representation of the expression.

        Returns
        -------
        str
            The human readable representation of the expression.
        """
        return str(self)


cpdef SymbolicExpr dimension(int32_t dim):
    r"""
    Constructs a ``SymbolicExpr`` representing coordinates of a dimension

    Parameters
    ----------
    dim : int
        Dimension index

    Returns
    -------
    SymbolicExpr
        A symbolic expression for the given dimension
    """
    cdef _SymbolicExpr handle

    with nogil:
        handle = _dimension(dim)
    return SymbolicExpr.from_handle(std_move(handle))


cpdef SymbolicExpr constant(int32_t value):
    r"""
    Constructs a ``SymbolicExpr`` representing a constant value

    Parameters
    ----------
    value : int
        Constant value to embed

    Returns
    -------
    SymbolicExpr
        A symbolic expression for the given constant
    """
    cdef _SymbolicExpr handle

    with nogil:
        handle = _constant(value)
    return SymbolicExpr.from_handle(std_move(handle))
