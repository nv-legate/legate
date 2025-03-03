# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int32_t, int64_t, uint64_t, uintptr_t
from libcpp cimport bool
from libcpp.utility cimport move as std_move
from libcpp.vector cimport vector as std_vector

from ...data_interface import Field, LegateDataInterfaceItem

from ..mapping.mapping cimport StoreTarget
from ..runtime.runtime cimport get_legate_runtime
from ..type.types cimport Type
from ..utilities.unconstructable cimport Unconstructable
from ..utilities.utils cimport is_iterable
from .logical_array cimport LogicalArray
from .physical_store cimport PhysicalStore
from .shape cimport Shape
from .slice cimport from_python_slice

from operator import index as operator_index

cdef class LogicalStore(Unconstructable):
    @staticmethod
    cdef LogicalStore from_handle(_LogicalStore handle):
        cdef LogicalStore result = LogicalStore.__new__(LogicalStore)
        result._handle = std_move(handle)
        # Enable out-of-order destruction, as we're in a GC language
        with nogil:
            result._handle.impl().get().allow_out_of_order_destruction()
        return result

    @property
    def shape(self) -> Shape:
        r"""
        Get the stores shape.

        :returns: The shape of the store.
        :rtype: Shape
        """
        cdef _Shape handle

        with nogil:
            handle = self._handle.shape()
        return Shape.from_handle(std_move(handle))

    @property
    def ndim(self) -> int32_t:
        r"""
        Get the number of dimensions of the store.

        :returns: The number of dimensions.
        :rtype: int
        """
        cdef int32_t ret

        with nogil:
            ret = self._handle.dim()
        return ret

    @property
    def has_scalar_storage(self) -> bool:
        r"""
        Get whether this store has been optimized for scalars.

        :returns: `True` if this store is backed by scalar storage, `False`
                 otherwise.
        :rtype: bool
        """
        cdef bool ret

        with nogil:
            ret = self._handle.has_scalar_storage()
        return ret

    cpdef bool overlaps(self, LogicalStore other):
        r"""
        Compute whether this store overlaps with another.

        Parameters
        ----------
        other : LogicalStore
            The store to compare against.

        Returns
        -------
        bool
            `True` if this store overlaps with `other`, `False` otherwise.
        """
        cdef bool ret

        with nogil:
            ret = self._handle.overlaps(other._handle)
        return ret

    @property
    def type(self) -> Type:
        r"""
        Get the type of the store.

        :rtype: Type
        :returns: The type of the store.
        """
        cdef _Type handle

        with nogil:
            handle = self._handle.type()
        return Type.from_handle(std_move(handle))

    @property
    def extents(self) -> tuple[int, ...]:
        r"""
        Get the extents of the store.

        This call may block if the store is unbound.

        :returns: The extents of the store.
        :rtype: tuple[int, ...]
        """
        cdef const std_vector[uint64_t] *ext = NULL

        with nogil:
            ext = &self._handle.extents().data()
        return tuple(ext[0])

    @property
    def volume(self) -> size_t:
        r"""
        Get the total number of elements in the store.

        This call may block if the store is unbound.

        :returns: The number of elements in the store.
        :rtype: int
        """
        cdef size_t ret

        with nogil:
            ret = self._handle.volume()
        return ret

    @property
    def size(self) -> size_t:
        r"""
        Get the total number of elements in the store.

        This property is an alias to `volume()`.

        :returns: The number of elements in the store.
        :rtype: int
        """
        return self.volume

    @property
    def unbound(self) -> bool:
        r"""
        Get whether the store is unbound.

        :returns: `True` if the store is unbound, `False` otherwise.
        :rtype: bool
        """
        cdef bool ret

        with nogil:
            ret = self._handle.unbound()
        return ret

    @property
    def transformed(self) -> bool:
        r"""
        Get whether the store is transformed.

        :returns: `True` if the store is transformed, `False` otherwise.
        :rtype: bool
        """
        cdef bool ret

        with nogil:
            ret = self._handle.transformed()
        return ret

    @property
    def __legate_data_interface__(self) -> LegateDataInterfaceItem:
        r"""
        Get the Legate array data interface.

        :returns: The array interface.
        :rtype: LegateDataInterfaceItem
        """
        array = LogicalArray.from_store(self)
        result: LegateDataInterfaceItem = {
            "version": 1,
            "data": {Field("store", self.type): array},
        }
        return result

    def __str__(self) -> str:
        r"""
        Return a human-readable representation of the store.

        Returns
        -------
        str
            The human readable representation of the store.
        """
        return self._handle.to_string().decode()

    def __repr__(self) -> str:
        r"""
        Return a human-readable representation of the store.

        Returns
        -------
        str
            The human readable representation of the store.
        """
        return str(self)

    def __getitem__(
        self, indices: int64_t | slice | tuple[int64_t | slice | None, ...],
    ) -> LogicalStore:
        r"""
        Get a sliced, projected, or promoted sub-store.

        Parameters
        ----------
        indices : int | slice | None | tuple[int | slice | None, ...]
            The indices to slice the current store with.

        Returns
        -------
        LogicalStore
            The transformed store.

        Notes
        -----
        For each value in `indices`, the store is transformed in the
        following manner:

        If `indices[i]` is a `slice`, the store is sliced:
        `store = store.slice(i, indices[i])`.

        If `indices[i]` is `None`, the store is promoted:
        `store = store.promote(i, 1)`.

        Otherwise, the store is projected:
        `store = store.project(i, indices[i])`.
        """
        cdef LogicalStore result = self

        if not isinstance(indices, tuple):
            indices = (indices,)

        cdef int dim

        for dim, index in enumerate(indices):
            if isinstance(index, slice):
                result = result.slice(dim, index)
            elif index is None:
                result = result.promote(dim, 1)
            else:
                result = result.project(dim, operator_index(index))

        return result

    cpdef LogicalStore promote(self, int32_t extra_dim, size_t dim_size):
        r"""
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

        cdef _LogicalStore handle

        with nogil:
            handle = self._handle.promote(extra_dim, dim_size)
        return LogicalStore.from_handle(std_move(handle))

    cpdef LogicalStore project(self, int32_t dim, int64_t index):
        r"""
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

        cdef _LogicalStore handle

        with nogil:
            handle = self._handle.project(dim, index)
        return LogicalStore.from_handle(std_move(handle))

    cpdef LogicalStore slice(self, int32_t dim, slice sl):
        r"""
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

        cdef _LogicalStore handle
        cdef _Slice cpp_slice = from_python_slice(sl)

        with nogil:
            handle = self._handle.slice(dim, std_move(cpp_slice))

        return LogicalStore.from_handle(std_move(handle))

    cpdef LogicalStore transpose(self, object axes):
        r"""
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

        cpp_axes.reserve(len(axes))
        for axis in axes:
            cpp_axes.push_back(axis)

        cdef _LogicalStore handle

        with nogil:
            handle = self._handle.transpose(std_move(cpp_axes))
        return LogicalStore.from_handle(std_move(handle))

    cpdef LogicalStore delinearize(self, int32_t dim, tuple shape):
        r"""
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
        cdef std_vector[uint64_t] sizes = std_vector[uint64_t]()

        sizes.reserve(len(shape))
        for value in shape:
            sizes.push_back(value)

        cdef _LogicalStore handle

        with nogil:
            handle = self._handle.delinearize(dim, std_move(sizes))
        return LogicalStore.from_handle(std_move(handle))

    cpdef void fill(self, object value):
        get_legate_runtime().issue_fill(self, value)

    cpdef LogicalStorePartition partition_by_tiling(self, object shape):
        r"""
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
        cdef std_vector[uint64_t] tile_shape = std_vector[uint64_t]()

        tile_shape.reserve(len(shape))
        for value in shape:
            tile_shape.push_back(value)

        cdef _LogicalStorePartition handle

        with nogil:
            handle = self._handle.partition_by_tiling(std_move(tile_shape))

        return LogicalStorePartition.from_handle(std_move(handle))

    cpdef PhysicalStore get_physical_store(self):
        r"""
        Get a `PhysicalStore` over this stores' data.

        Returns
        -------
        PhysicalStore
            The `PhysicalStore` spanning this stores' data.
        """
        cdef _PhysicalStore handle

        with nogil:
            handle = self._handle.get_physical_store()
        return PhysicalStore.from_handle(std_move(handle))

    cpdef void detach(self):
        r"""
        Detach a store from its attached memory.

        This call will wait for all operations that use the store (or any
        sub-store) to complete.

        After this call returns, it is safe to deallocate the attached
        external allocation. If the allocation was mutable, the contents
        would be up-to-date upon the return. The contents of the store are
        invalid after that point.
        """
        with nogil:
            self._handle.detach()

    cpdef void offload_to(self, StoreTarget target_mem):
        r"""
        Offload store to specified target memory. This call copies the store to
        the specified target memory and makes the copy exclusive to that
        memory, thus allowing the runtime to discard any other copies and make
        space in other memories.

        Parameters
        ----------
        target_mem : StoreTarget
            The target memory to offload to

        """
        with nogil:
            self._handle.offload_to(target_mem)

    cpdef bool equal_storage(self, LogicalStore other):
        r"""
        Determine whether two stores refer to the same memory.

        This routine can be used to determine whether two seemingly unrelated
        stores refer to the same logical memory region, including through
        possible transformations in either `self` or `other`.

        The user should note that some transformations *do* modify the
        underlying storage. For example, the store produced by slicing will
        *not* share the same storage as its parent, and this routine will
        return false for it.

        Transposed stores, on the other hand, still share the same storage,
        and hence this routine will return true for them.

        Parameters
        ----------
        other : LogicalStore
            The `LogicalStore` against which to compare.

        Returns
        -------
        bool
            `True` if this store overlaps with `other`, `False` otherwise.
        """
        cdef bool ret

        with nogil:
            ret = self._handle.equal_storage(other._handle)
        return ret

    @property
    def raw_handle(self) -> uintptr_t:
        r"""
        Get a handle to the C++ ``LogicalStore`` object. This property is an
        escape-hatch that exists out of necessity. We make no guarantees about
        its type, behavior, or other properties except for the fact that it
        represents an opaque handle to the underlying C++ object.

        Currently, it returns the raw pointer to the C++ object.

        :returns: A handle to the C++ object.
        :rtype: int
        """
        return <uintptr_t> &self._handle


cdef class LogicalStorePartition:
    @staticmethod
    cdef LogicalStorePartition from_handle(_LogicalStorePartition handle):
        cdef LogicalStorePartition result = LogicalStorePartition.__new__(
            LogicalStorePartition
        )
        result._handle = std_move(handle)
        return result

    cpdef LogicalStore store(self):
        r"""
        Get the `LogicalStore` to which this partition refers to.

        Returns
        -------
        LogicalStore
            The `LogicalStore`.
        """
        cdef _LogicalStore handle

        with nogil:
            handle = self._handle.store()
        return LogicalStore.from_handle(std_move(handle))

    @property
    def color_shape(self) -> tuple[int, ...]:
        r"""
        Get the color shape for this partition.

        :returns: The color shape for this partition.
        :rtype: tuple[int, ...]
        """
        cdef const std_vector[uint64_t] *v = NULL

        with nogil:
            v = &self._handle.color_shape().data()
        return tuple(v[0])

    def get_child_store(self, *color) -> LogicalStore:
        r"""
        Get a child store from this partition.

        Parameters
        ----------
        *color : int
            The colors of the child stores to get.

        Returns
        -------
        LogicalStore
            The `LogicalStore` comprising the selected children.
        """
        cdef _tuple[uint64_t] cpp_color

        cpp_color.reserve(len(color))
        for coord in color:
            cpp_color.append_inplace(<uint64_t> coord)

        cdef _LogicalStore handle

        with nogil:
            handle = self._handle.get_child_store(std_move(cpp_color))
        return LogicalStore.from_handle(std_move(handle))
