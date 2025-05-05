# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int32_t
from libcpp.utility cimport move as std_move

from .inline_allocation cimport _InlineAllocation, InlineAllocation

from ..type.types cimport Type

from typing import Any


cdef class TaskLocalBuffer(Unconstructable):
    @staticmethod
    cdef TaskLocalBuffer from_handle(
        const _TaskLocalBuffer &handle, object owner = None
    ):
        cdef TaskLocalBuffer result = TaskLocalBuffer.__new__(
            TaskLocalBuffer
        )

        result._handle = handle
        result._alloc = None
        result._owner = owner
        return result

    @property
    def type(self) -> Type:
        r"""
        :returns: The type of the buffer.
        :rtype: Type
        """
        cdef _Type ty

        with nogil:
            ty = self._handle.type()

        return Type.from_handle(std_move(ty))

    @property
    def dim(self) -> int32_t:
        r"""
        :returns: The dimension of the buffer.
        :rtype: int
        """
        cdef int32_t dim

        with nogil:
            dim = self._handle.dim()

        return dim

    cdef InlineAllocation _init_inline_allocation(self):
        cdef _InlineAllocation alloc

        with nogil:
            alloc = self._handle.get_inline_allocation()

        cdef tuple strides
        cdef tuple shape

        if self.dim == 0:
            strides = ()
            shape = ()
        else:
            strides = tuple(alloc.strides)
            shape = InlineAllocation._compute_shape(self._handle.domain())

        return InlineAllocation.create(
            handle=std_move(alloc),
            ty=self.type,
            shape=shape,
            strides=strides,
            owner=self
        )

    cdef InlineAllocation _get_inline_allocation(self):
        if self._alloc is None:
            self._alloc = self._init_inline_allocation()
        return self._alloc

    @property
    def __array_interface__(self) -> dict[str, Any]:
        r"""
        Retrieve the numpy-compatible array representation of the buffer.

        :returns: The numpy array interface dict.
        :rtype: dict[str, Any]
        """
        return self._get_inline_allocation().__array_interface__

    @property
    def __cuda_array_interface__(self) -> dict[str, Any]:
        r"""
        Retrieve the cupy-compatible array representation of the buffer.

        :returns: The cupy array interface dict.
        :rtype: dict[str, Any]
        """
        return self._get_inline_allocation().__cuda_array_interface__
