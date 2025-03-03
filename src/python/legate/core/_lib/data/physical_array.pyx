# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint32_t

from ..type.types cimport Type
from ..utilities.unconstructable cimport Unconstructable
from .physical_store cimport PhysicalStore


cdef class PhysicalArray(Unconstructable):
    @staticmethod
    cdef PhysicalArray from_handle(const _PhysicalArray &array):
        cdef PhysicalArray result = PhysicalArray.__new__(PhysicalArray)
        result._handle = array
        return result

    @property
    def nullable(self) -> bool:
        r"""
        Get whether this array is nullable.

        :returns: `True` if the array is nullable, `False` otherwise.
        :rtype: bool
        """
        return self._handle.nullable()

    @property
    def ndim(self) -> int:
        r"""
        Get the number of dimensions in the array.

        :returns: The number of dimensions in the array.
        :rtype: int
        """
        return self._handle.dim()

    @property
    def type(self) -> Type:
        r"""
        Get the type of the array.

        :returns: The type of the array.
        :rtype: Type
        """
        return Type.from_handle(self._handle.type())

    @property
    def nested(self) -> bool:
        r"""
        Get whether this array has child arrays.

        :returns: `True` if this array has children, `False` otherwise.
        :rtype: bool
        """
        return self._handle.nested()

    cpdef PhysicalStore data(self):
        r"""
        Get the store containing the array's data.

        Returns
        -------
        PhysicalStore
            The store containing the array's data.
        """
        return PhysicalStore.from_handle(self._handle.data())

    cpdef PhysicalStore null_mask(self):
        r"""
        Get the store containing the array's null mask.

        Returns
        -------
        PhysicalStore
            The store containing the array's null mask.
        """
        return PhysicalStore.from_handle(self._handle.null_mask())

    cpdef PhysicalArray child(self, uint32_t index):
        r"""
        Get the sub-array of a given index.

        Parameters
        ----------
        index : int
            The child index which to get.

        Returns
        -------
        PhysicalArray
            The child array.
        """
        return PhysicalArray.from_handle(self._handle.child(index))

    cpdef Domain domain(self):
        r"""
        Get the `Domain` of the array.

        Returns
        -------
        Domain
            The array's domain.
        """
        return Domain.from_handle(self._handle.domain())

    @property
    def __array_interface__(self) -> dict[str, Any]:
        r"""
        Retrieve the numpy-compatible array representation of the allocation.

        :returns: The numpy array interface dict.
        :rtype: dict[str, Any]

        :raises ValueError: If the array is nullable or nested.
        """
        if self.nullable or self.nested:
            raise ValueError(
                "Nested or nullable arrays don't support the array interface "
                "directly"
            )
        return self.data().__array_interface__

    @property
    def __cuda_array_interface__(self) -> dict[str, Any]:
        r"""
        Retrieve the cupy-compatible array representation of the allocation.

        :returns: The cupy array interface dict.
        :rtype: dict[str, Any]

        :raises ValueError: If the array is nullable or nested.
        """
        if self.nullable or self.nested:
            raise ValueError(
                "Nested or nullable arrays don't support the CUDA array "
                "interface directly"
            )
        return self.data().__cuda_array_interface__
