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

from cython.operator cimport dereference
from libc.stdint cimport int32_t, int64_t, uint32_t, uint64_t, uintptr_t
from libcpp.utility cimport move as std_move
from libcpp.vector cimport vector as std_vector

from ...data_interface import Field, LegateDataInterfaceItem

from ..runtime.runtime cimport get_legate_runtime
from ..type.type_info cimport Type
from ..utilities.unconstructable cimport Unconstructable
from ..utilities.utils cimport is_iterable
from .logical_store cimport LogicalStore
from .shape cimport Shape
from .slice cimport from_python_slice


cdef class LogicalArray(Unconstructable):
    @staticmethod
    cdef LogicalArray from_handle(_LogicalArray handle):
        cdef LogicalArray result = LogicalArray.__new__(LogicalArray)
        result._handle = handle
        return result

    @staticmethod
    def from_store(LogicalStore store) -> LogicalArray:
        r"""
        Create a `LogicalArray` from a `LogicalStore`.

        Parameters
        ----------
        store : LogicalStore
            The store to create the array from.

        Returns
        -------
        LogicalArray
            The newly created `LogicalArray`.
        """
        return LogicalArray.from_handle(_LogicalArray(store._handle))

    @staticmethod
    def from_raw_handle(uintptr_t raw_handle):
        r"""
        Create a `LogicalArray` from a pointer to C++ `LogicalArray`.

        Parameters
        ----------
        raw_handle : int
            The pointer to the C++ `LogicalArray` object.

        Returns
        -------
        LogicalArray
            The newly created `LogicalArray`.
        """
        return LogicalArray.from_handle(
            dereference(<_LogicalArray*> raw_handle)
        )

    @property
    def shape(self) -> Shape:
        r"""
        Get the shape of the array.

        Returns
        -------
        Shape
            The shape of the array.
        """
        return Shape.from_handle(self._handle.shape())

    @property
    def ndim(self) -> int32_t:
        r"""
        Get the dimension of the array.

        Returns
        -------
        int
            The dimension of the array.
        """
        return self._handle.dim()

    @property
    def type(self) -> Type:
        r"""
        Get the type of the array.

        Returns
        -------
        Type
            The `Type` of the array.
        """
        return Type.from_handle(self._handle.type())

    @property
    def extents(self) -> tuple[uint64_t, ...]:
        r"""
        Get the extents of the array.

        Returns
        -------
        tuple[int, ...]
            The extents of the array.
        """
        return self._handle.extents().data()

    @property
    def volume(self) -> size_t:
        r"""
        Get the volume of the array.

        Returns
        -------
        int
            The volume of the array.
        """
        return self._handle.volume()

    @property
    def size(self) -> size_t:
        r"""
        Get the size of the array. This method is an alias of `volume()`.

        Returns
        -------
        int
            The size of the array.
        """
        return self.volume

    @property
    def unbound(self) -> bool:
        r"""
        Return whether the store is unbound or not.

        Returns
        -------
        bool
            `True` if the array is unbound, `False` otherwise.
        """
        return self._handle.unbound()

    @property
    def nullable(self) -> bool:
        r"""
        Get whether this array is nullable.

        Returns
        -------
        bool
            `True` if this array is nullable, `False` otherwise.
        """
        return self._handle.nullable()

    @property
    def nested(self) -> bool:
        r"""
        Get whether this array is nested

        Returns
        -------
        bool
            `True` if this array has nested stores, `False` otherwise.
        """
        return self._handle.nested()

    @property
    def num_children(self) -> uint32_t:
        r"""
        Get the number of child stores of this array.

        Returns
        -------
        int
            The number of child stores of the array.
        """
        return self._handle.num_children()

    @property
    def __legate_data_interface__(self) -> LegateDataInterfaceItem:
        r"""
        Get the legate data interface of the array.

        Returns
        -------
        LegateDataInterfaceItem
            The interface to the array.
        """
        result: LegateDataInterfaceItem = {
            "version": 1,
            "data": {Field("store", self.type): self},
        }
        return result

    def __getitem__(
        self, indices: int64_t | slice | tuple[int64_t | slice, ...],
    ) -> LogicalArray:
        r"""
        Get a subarray of this array.

        Parameters
        ----------
        indices : int | slice | tuple[int | slice, ...]
            The index or slice(s) of indices of the subarrays to retrieve.

        Returns
        -------
        LogicalArray
            The array comprising the subarrays of the slice.

        Raises
        ------
        NotImplementedError
            If the array is nested or nullable.
        """
        if self.nested or self.nullable:
            raise NotImplementedError(
                "Indexing is not implemented for nested or nullable arrays"
            )

        return LogicalArray.from_store(self.data[indices])

    cpdef LogicalArray promote(self, int32_t extra_dim, size_t dim_size):
        r"""
        Add an extra dimension to the array. This call may block if the array
        is unbound.

        Parameters
        ----------
        extra_dim : int
            Position for the extra dimension.
        dim_size : int
            Size of the new dimension to add.

        Returns
        -------
        LogicalArray
            The promoted array.
        """
        return LogicalArray.from_handle(
            self._handle.promote(extra_dim, dim_size)
        )

    cpdef LogicalArray project(self, int32_t dim, int64_t index):
        r"""
        Project out or "flatten" a dimension of the array. This call may block
        if the array is unbound.

        Parameters
        ----------
        dim : int
            The dimension to project out.
        index : int
            The index of the chosen dimension.

        Returns
        -------
        LogicalArray
            The array with the projected-out dimension.
        """
        return LogicalArray.from_handle(self._handle.project(dim, index))

    cpdef LogicalArray slice(self, int32_t dim, slice sl):
        r"""
        Get a contiguous sub-section of the array. This call may block if the
        array is unbound.

        Parameters
        ----------
        dim : int
            Dimension to slice at.
        sl : slice
            The slice descriptor

        Returns
        -------
        LogicalArray
            The array comprising the sliced section.
        """
        return LogicalArray.from_handle(
            self._handle.slice(dim, from_python_slice(sl))
        )

    cpdef LogicalArray transpose(self, object axes):
        r"""
        Get a transpose of the array. This call may block if the array is
        unbound.

        Parameters
        ----------
        axes : Collection[int]
            A mapping from dimension of the input array to dimension of the
            output array.

        Returns
        -------
        LogicalArray
            The transposed array.

        Raises
        ------
        ValueError
            If `axes` is not iterable.
        """
        if not is_iterable(axes):
            raise ValueError(f"Expected an iterable but got {type(axes)}")
        cdef std_vector[int32_t] cpp_axes = std_vector[int32_t]()

        cpp_axes.reserve(len(axes))
        for axis in axes:
            cpp_axes.push_back(axis)
        return LogicalArray.from_handle(
            self._handle.transpose(std_move(cpp_axes))
        )

    cpdef LogicalArray delinearize(self, int32_t dim, object shape):
        r"""
        Delinearize a dimension into multiple dimensions. This call may block
        if the array is unbound.

        Parameters
        ----------
        dim : int
            Dimension to delinearize.
        shape : Collection[int]
            The extents for the resulting dimensions

        Returns
        -------
        LogicalArray
            The array with the chosen delinearized dimension.

        Raises
        ------
        ValueError
            If `shape` is not iterable.
        """
        if not is_iterable(shape):
            raise ValueError(f"Expected an iterable but got {type(shape)}")
        cdef std_vector[uint64_t] sizes = std_vector[uint64_t]()

        sizes.reserve(len(shape))
        for value in shape:
            sizes.push_back(value)
        return LogicalArray.from_handle(
            self._handle.delinearize(dim, std_move(sizes))
        )

    cpdef void fill(self, object value):
        r"""
        Fill the array with a value.

        Parameters
        ----------
        value : Any
            The value to fill the array with.
        """
        get_legate_runtime().issue_fill(self, value)

    @property
    def data(self) -> LogicalStore:
        r"""
        Get the `LogicalStore` of this array.

        Returns
        -------
        LogicalStore
            The `LogicalStore` of this array.
        """
        return LogicalStore.from_handle(self._handle.data())

    @property
    def null_mask(self) -> LogicalStore:
        r"""
        Get the null mask of this array.

        Returns
        -------
        LogicalStore
            The null mask of this array.
        """
        return LogicalStore.from_handle(self._handle.null_mask())

    cpdef LogicalArray child(self, uint32_t index):
        r"""
        Get a child array of this array.

        Parameters
        ----------
        index : int
            The index of the sub array.

        Returns
        -------
        LogicalArray
            The sub array.
        """
        return LogicalArray.from_handle(self._handle.child(index))

    cpdef PhysicalArray get_physical_array(self):
        r"""
        Get a `PhysicalArray` of the data for this array.

        Returns
        -------
        PhysicalArray
            The physical array.
        """
        cdef _PhysicalArray array
        with nogil:
            array = self._handle.get_physical_array()

        return PhysicalArray.from_handle(array)

    @property
    def raw_handle(self) -> uintptr_t:
        r"""
        Get a pointer to the C++ `LogicalArray` object.

        Returns
        -------
        int
            The pointer to the C++ `LogicalArray` object.
        """
        return <uintptr_t> &self._handle


cdef _LogicalArray to_cpp_logical_array(object array_or_store):
    if isinstance(array_or_store, LogicalArray):
        return (<LogicalArray> array_or_store)._handle
    if isinstance(array_or_store, LogicalStore):
        return _LogicalArray((<LogicalStore> array_or_store)._handle)
    raise ValueError(
        "Expected a logical array or store but got "
        f"{type(array_or_store)}"
    )
