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

from libc.stdint cimport int32_t, int64_t, uintptr_t
from libcpp cimport bool
from libcpp.utility cimport move as std_move
from libcpp.vector cimport vector as std_vector

from ...data_interface import Field, LegateDataInterfaceItem
from ...shape import Shape
from ...utils import is_iterable

from ..type.type_info cimport Type
from .physical_store cimport PhysicalStore
from .slice cimport from_python_slice


cdef class LogicalStore:
    @staticmethod
    cdef LogicalStore from_handle(_LogicalStore handle):
        cdef LogicalStore result = LogicalStore.__new__(LogicalStore)
        result._handle = handle
        # Enable out-of-order destruction, as we're in a GC language
        handle.impl().get().allow_out_of_order_destruction()
        return result

    @property
    def shape(self) -> Shape:
        return Shape(self.extents)

    @property
    def ndim(self) -> int32_t:
        return self._handle.dim()

    @property
    def has_scalar_storage(self) -> bool:
        return self._handle.has_scalar_storage()

    def overlaps(self, LogicalStore other) -> bool:
        return self._handle.overlaps(other._handle)

    @property
    def type(self) -> Type:
        return Type.from_handle(self._handle.type())

    @property
    def extents(self) -> list:
        return self._handle.extents().data()

    @property
    def volume(self) -> size_t:
        return self._handle.volume()

    @property
    def size(self) -> size_t:
        return self.volume

    @property
    def unbound(self) -> bool:
        return self._handle.unbound()

    @property
    def transformed(self) -> bool:
        return self._handle.transformed()

    @property
    def __legate_data_interface__(self) -> LegateDataInterfaceItem:
        from .logical_array import LogicalArray

        array = LogicalArray.from_store(self)
        result: LegateDataInterfaceItem = {
            "version": 1,
            "data": {Field("store", self.type): array},
        }
        return result

    def __str__(self) -> str:
        return self._handle.to_string().decode()

    def __repr__(self) -> str:
        return str(self)

    def promote(self, int32_t extra_dim, size_t dim_size) -> LogicalStore:
        """
        Adds an extra dimension to the store. Value of ``extra_dim`` decides
        where a new dimension should be added, and each dimension `i`, where
        `i` >= ``extra_dim``, is mapped to dimension `i+1` in a returned store.
        A returned store provides a view to the input store where the values
        are broadcasted along the new dimension.

        For example, for a 1D store ``A`` contains ``[1, 2, 3]``,
        ``A.promote(0, 2)`` yields a store equivalent to:

        ::

            [[1, 2, 3],
             [1, 2, 3]]

        whereas ``A.promote(1, 2)`` yields:

        ::

            [[1, 1],
             [2, 2],
             [3, 3]]

        Parameters
        ----------
        extra_dim : int
            Position for a new dimension
        dim_size : int, optional
            Extent of the new dimension

        Returns
        -------
        LogicalStore
            A new store with an extra dimension

        Raises
        ------
        ValueError
            If ``extra_dim`` is not a valid dimension name
        """
        if extra_dim < 0:
            extra_dim += self.ndim
        return LogicalStore.from_handle(
            self._handle.promote(extra_dim, dim_size)
        )

    def project(self, int32_t dim, int64_t index) -> LogicalStore:
        """
        Projects out a dimension of the store. Each dimension `i`, where
        `i` > ``dim``, is mapped to dimension `i-1` in a returned store.
        A returned store provides a view to the input store where the values
        are on hyperplane :math:`x_\\mathtt{dim} = \\mathtt{index}`.

        For example, if a 2D store ``A`` contains ``[[1, 2], [3, 4]]``,
        ``A.project(0, 1)`` yields a store equivalent to ``[3, 4]``, whereas
        ``A.project(1, 0)`` yields ``[1, 3]``.

        Parameters
        ----------
        dim : int
            Dimension to project out
        index : int
            Index on the chosen dimension

        Returns
        -------
        LogicalStore
            A new store with one fewer dimension

        Raises
        ------
        ValueError
            If ``dim`` is not a valid dimension name or ``index`` is
            out of bounds
        """
        if dim < 0:
            dim += self.ndim
        return LogicalStore.from_handle(self._handle.project(dim, index))

    def slice(self, int32_t dim, slice sl) -> LogicalStore:
        """
        Slices a contiguous sub-section of the store.

        For example, consider a 2D store ``A``

        ::

            [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]]

        A slicing ``A.slice(0, slice(1, None))`` yields:

        ::

            [[4, 5, 6],
             [7, 8, 9]]

        The result store will look like this on a different slicing call
        ``A.slice(1, slice(None, 2))``:

        ::

            [[1, 2],
             [4, 5],
             [7, 8]]

        Finally, chained slicing calls

        ::

            A.slice(0, slice(1, None)).slice(1, slice(None, 2))

        results in:

        ::

            [[4, 5],
             [7, 8]]


        Parameters
        ----------
        dim : int
            Dimension to slice
        sl : slice
            Slice that expresses a sub-section

        Returns
        -------
        LogicalStore
            A new store that correponds to the sliced section

        Notes
        -----
        Slicing with a non-unit step is currently not supported.

        Raises
        ------
        ValueError
            If ``sl.step`` is not a unit or ``sl`` is out of bounds
        """
        if dim < 0:
            dim += self.ndim
        return LogicalStore.from_handle(
            self._handle.slice(dim, from_python_slice(sl))
        )

    def transpose(self, object axes) -> LogicalStore:
        """
        Reorders dimensions of the store. Dimension ``i`` of the resulting
        store is mapped to dimension ``axes[i]`` of the input store.

        For example, for a 3D store ``A``

        ::

            [[[1, 2],
              [3, 4]],

             [[5, 6],
              [7, 8]]]

        transpose calls ``A.transpose([1, 2, 0])`` and ``A.transpose([2, 1,
        0])`` yield the following stores, respectively:

        ::

            [[[1, 5],
              [2, 6]],

             [[3, 7],
              [4, 8]]]


        ::

            [[[1, 5],
              [3, 7]],

             [[2, 6],
              [4, 8]]]


        Parameters
        ----------
        axes : tuple[int]
            Mapping from dimensions of the resulting store to those of the
            input

        Returns
        -------
        LogicalStore
            A new store with the dimensions transposed

        Raises
        ------
        ValueError
            If any of the following happens: 1) The length of ``axes`` doesn't
            match the store's dimension; 2) ``axes`` has duplicates; 3) Any
            value in ``axes`` is negative, or greater than or equal to the
            store's dimension
        """
        if not is_iterable(axes):
            raise ValueError(f"Expected an iterable but got {type(axes)}")
        cdef std_vector[int32_t] cpp_axes = std_vector[int32_t]()
        for axis in axes:
            cpp_axes.push_back(axis)
        return LogicalStore.from_handle(
            self._handle.transpose(std_move(cpp_axes))
        )

    def delinearize(self, int32_t dim, tuple shape) -> LogicalStore:
        """
        Delinearizes a dimension into multiple dimensions. Each dimension
        `i` of the store, where `i` > ``dim``, will be mapped to dimension
        `i+N` of the resulting store, where `N` is the length of ``shape``.
        A delinearization that does not preserve the size of the store is
        invalid.

        For example, consider a 2D store ``A``

        ::

            [[1, 2, 3, 4],
             [5, 6, 7, 8]]

        A delinearizing call `A.delinearize(1, [2, 2]))` yields:

        ::

            [[[1, 2],
              [3, 4]],

             [[5, 6],
              [7, 8]]]

        Parameters
        ----------
        dim : int
            Dimension to delinearize
        shape : tuple[int]
            New shape for the chosen dimension

        Returns
        -------
        LogicalStore
            A new store with the chosen dimension delinearized

        Notes
        -----
        Unlike other transformations, delinearization is not an affine
        transformation. Due to this nature, delinearized stores can raise
        `NonInvertibleError` in places where they cannot be used.

        Raises
        ------
        ValueError
            If ``dim`` is invalid for the store or ``shape`` does not preserve
            the size of the chosen dimenison
        """
        if dim < 0:
            dim += self.ndim
        cdef std_vector[int64_t] sizes = std_vector[int64_t]()
        for value in shape:
            sizes.push_back(value)
        return LogicalStore.from_handle(
            self._handle.delinearize(dim, std_move(sizes))
        )

    def partition_by_tiling(self, object shape) -> LogicalStorePartition:
        """
        Creates a tiled partition of the store

        Parameters
        ----------
        tile_shape : tuple[int]
            Shape of tiles

        Returns
        -------
        LogicalStorePartition
            A ``LogicalStorePartition`` object
        """
        if not is_iterable(shape):
            raise ValueError(f"Expected an iterable but got {type(shape)}")
        cdef std_vector[size_t] tile_shape = std_vector[size_t]()
        for value in shape:
            tile_shape.push_back(value)
        return LogicalStorePartition.from_handle(
            self._handle.partition_by_tiling(std_move(tile_shape))
        )

    def get_physical_store(self) -> PhysicalStore:
        return PhysicalStore.from_handle(self._handle.get_physical_store())

    def detach(self) -> None:
        self._handle.detach()

    @property
    def raw_handle(self) -> uintptr_t:
        return <uintptr_t> &self._handle


cdef class LogicalStorePartition:
    @staticmethod
    cdef LogicalStorePartition from_handle(_LogicalStorePartition handle):
        cdef LogicalStorePartition result = LogicalStorePartition.__new__(
            LogicalStorePartition
        )
        result._handle = handle
        return result

    def store(self) -> LogicalStore:
        return LogicalStore.from_handle(self._handle.store())

    @property
    def color_shape(self) -> list:
        return self._handle.color_shape().data()
