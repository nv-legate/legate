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

from cython.operator cimport dereference
from libc.stdint cimport int32_t, int64_t, uint32_t, uintptr_t
from libcpp.utility cimport move as std_move
from libcpp.vector cimport vector as std_vector

from ..type.type_info cimport Type
from .logical_store cimport LogicalStore
from .slice cimport from_python_slice

from ...shape import Shape
from ...utils import is_iterable


cdef class LogicalArray:
    @staticmethod
    cdef LogicalArray from_handle(_LogicalArray handle):
        cdef LogicalArray result = LogicalArray.__new__(LogicalArray)
        result._handle = handle
        return result

    @staticmethod
    def from_store(LogicalStore store) -> LogicalArray:
        return LogicalArray.from_handle(_LogicalArray(store._handle))

    @staticmethod
    def from_raw_handle(uintptr_t raw_handle):
        return LogicalArray.from_handle(dereference(<_LogicalArray*> raw_handle))

    @property
    def shape(self) -> Shape:
        return Shape(self.extents)

    @property
    def ndim(self) -> int32_t:
        return self._handle.dim()

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
    def nullable(self) -> bool:
        return self._handle.nullable()

    @property
    def nested(self) -> bool:
        return self._handle.nested()

    @property
    def num_children(self) -> uint32_t:
        return self._handle.num_children()

    def promote(self, int32_t extra_dim, size_t dim_size) -> LogicalArray:
        return LogicalArray.from_handle(
            self._handle.promote(extra_dim, dim_size)
        )

    def project(self, int32_t dim, int64_t index) -> LogicalArray:
        return LogicalArray.from_handle(self._handle.project(dim, index))

    def slice(self, int32_t dim, slice sl) -> LogicalArray:
        return LogicalArray.from_handle(
            self._handle.slice(dim, from_python_slice(sl))
        )

    def transpose(self, object axes) -> LogicalArray:
        if not is_iterable(axes):
            raise ValueError(f"Expected an iterable but got {type(axes)}")
        cdef std_vector[int32_t] cpp_axes = std_vector[int32_t]()
        for axis in axes:
            cpp_axes.push_back(axis)
        return LogicalArray.from_handle(
            self._handle.transpose(std_move(cpp_axes))
        )

    def delinearize(self, int32_t dim, object shape) -> LogicalArray:
        if not is_iterable(shape):
            raise ValueError(f"Expected an iterable but got {type(shape)}")
        cdef std_vector[uint64_t] sizes = std_vector[uint64_t]()
        sizes.reserve(len(shape))
        for value in shape:
            sizes.push_back(value)
        return LogicalArray.from_handle(
            self._handle.delinearize(dim, std_move(sizes))
        )

    @property
    def data(self) -> LogicalStore:
        return LogicalStore.from_handle(self._handle.data())

    @property
    def null_mask(self) -> LogicalStore:
        return LogicalStore.from_handle(self._handle.null_mask())

    def child(self, uint32_t index) -> LogicalArray:
        return LogicalArray.from_handle(self._handle.child(index))

    def raw_handle(self) -> uintptr_t:
        return <uintptr_t> &self._handle


cdef _LogicalArray to_cpp_logical_array(object array_or_store):
    cdef _LogicalArray result
    if isinstance(array_or_store, LogicalArray):
        result = (<LogicalArray> array_or_store)._handle
    elif isinstance(array_or_store, LogicalStore):
        result = _LogicalArray((<LogicalStore> array_or_store)._handle)
    else:
        raise ValueError(
            "Expected a logical array or store but got "
            f"{type(array_or_store)}"
        )
    return result
