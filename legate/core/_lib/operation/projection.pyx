# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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
        return self._handle.dim()

    @property
    def weight(self) -> int:
        r"""
        Get the weight for the coordinates.

        :returns: The weight.
        :rtype: int
        """
        return self._handle.weight()

    @property
    def offset(self) -> int:
        r"""
        Get the offset of the expression.

        :returns: The offset.
        :rtype: int
        """
        return self._handle.offset()

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
        return self._handle.is_identity(dim)

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
        return SymbolicExpr.from_handle(self._handle * other)

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
        return SymbolicExpr.from_handle(self._handle + other)

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
    """
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
    return SymbolicExpr.from_handle(_dimension(dim))


cpdef SymbolicExpr constant(int32_t value):
    """
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
    return SymbolicExpr.from_handle(_constant(value))
