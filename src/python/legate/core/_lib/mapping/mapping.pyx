# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp.vector cimport vector as std_vector
from libc.stdint cimport int32_t

from ..utilities.unconstructable cimport Unconstructable

cdef class DimOrdering(Unconstructable):
    Kind = DimOrderingKind

    @staticmethod
    cdef DimOrdering from_handle(_DimOrdering handle):
        """Create a DimOrdering instance from a C++ handle.

        Parameters
        ----------
        handle : _DimOrdering
            The C++ DimOrdering handle

        Returns
        -------
        DimOrdering
            A new DimOrdering instance
        """
        cdef DimOrdering result = DimOrdering.__new__(DimOrdering)
        result._handle = handle
        return result

    @staticmethod
    def c_order() -> DimOrdering:
        """Create a C-order dimension ordering.

        Returns
        -------
        DimOrdering
            A DimOrdering instance with C-order (row-major) layout
        """
        return DimOrdering.from_handle(_DimOrdering.c_order())

    @staticmethod
    def fortran_order() -> DimOrdering:
        """Create a Fortran-order dimension ordering.

        Returns
        -------
        DimOrdering
            A DimOrdering instance with Fortran-order (column-major) layout
        """
        return DimOrdering.from_handle(_DimOrdering.fortran_order())

    @staticmethod
    def custom_order(dims: list[int]) -> DimOrdering:
        """Create a custom dimension ordering.

        Parameters
        ----------
        dims : list of int
            List of dimension indices specifying the custom ordering

        Returns
        -------
        DimOrdering
            A DimOrdering instance with the specified custom ordering
        """
        cdef std_vector[int32_t] cpp_dims
        for dim in dims:
            cpp_dims.push_back(dim)
        return DimOrdering.from_handle(_DimOrdering.custom_order(cpp_dims))

    @property
    def kind(self):
        """Returns the kind of dimension ordering (C, FORTRAN, or CUSTOM)

        Returns
        -------
        DimOrdering.Kind
            The kind of dimension ordering
        """
        return self._handle.kind()
