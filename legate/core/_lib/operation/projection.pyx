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

from libc.stdint cimport int32_t
from libcpp.utility cimport move as std_move


cdef class SymbolicExpr:
    @staticmethod
    cdef SymbolicExpr from_handle(_SymbolicExpr handle):
        cdef SymbolicExpr result = SymbolicExpr.__new__(SymbolicExpr)
        result._handle = std_move(handle)
        return result

    @property
    def dim(self) -> int:
        return self._handle.dim()

    @property
    def weight(self) -> int:
        return self._handle.weight()

    @property
    def offset(self) -> int:
        return self._handle.offset()

    cpdef bool is_identity(self, int32_t dim):
        return self._handle.is_identity(dim)

    def __eq__(self, object other) -> bool:
        if not isinstance(other, SymbolicExpr):
            return False
        return self._handle == (<SymbolicExpr> other)._handle

    def __mul__(self, int32_t other) -> SymbolicExpr:
        return SymbolicExpr.from_handle(self._handle * other)

    def __add__(self, int32_t other) -> SymbolicExpr:
        return SymbolicExpr.from_handle(self._handle + other)

    def __str__(self) -> str:
        return self._handle.to_string().decode()

    def __repr__(self) -> str:
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
