# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import math

import numpy as np

from libc.stdint cimport int32_t, uintptr_t
from cpython cimport Py_buffer, PyObject_GetBuffer

from ..type.types cimport Type
from ..utilities.typedefs cimport _Domain, _DomainPoint
from ..mapping.mapping cimport StoreTarget
from .physical_store cimport PhysicalStore

# This object exists purely to circumvent numpy. When you do
#
# arr = np.asarray(obj)
#
# Where obj implements both `__array_interface__` and `__getbuffer__`, then
# numpy will prefer `__getbuffer__` (seemingly because it is the "standard" way
# of doing things).
#
# But we implement our `__getbuffer__` in terms of `__array_interface__` by
# creating a numpy array from ourselves and calling PyObject_GetBuffer() on
# it. This is done for convenience since implementing `__getbuffer__` is kind
# of complicated:
#
# def __getbuffer__(self, ...):
#     self_np = np.asarray(self)
#     PyObject_GetBuffer(self_np, ...)
#
# But this leads to infinite recursion since `np.asarray()` will just keep
# calling `__getbuffer__`. So we need an in-between that exposes only the numpy
# array interface but not the buffer protocol.
cdef class _OnlyArrayInterface:
    cdef InlineAllocation alloc

    def __init__(self, InlineAllocation alloc) -> None:
        self.alloc = alloc

    @property
    def __array_interface__(self):
        return self.alloc.__array_interface__


cdef class InlineAllocation:
    @staticmethod
    cdef InlineAllocation create(
        PhysicalStore store, _InlineAllocation handle
    ):
        cdef InlineAllocation result = InlineAllocation.__new__(
            InlineAllocation
        )
        result._handle = handle
        result._store = store
        result._shape = None
        return result

    @property
    def ptr(self) -> uintptr_t:
        r"""
        Access the raw pointer to the allocation.

        :returns: The raw pointer to the allocation.
        :rtype: int
        """
        return <uintptr_t>(self._handle.ptr)

    @property
    def strides(self) -> tuple[size_t, ...]:
        r"""
        Retrieve the strides of the allocation.

        If the allocation has dimension 0, an empty tuple is returned.

        :returns: The strides of the allocation.
        :rtype: tuple[int, ...]
        """
        return () if self._store.ndim == 0 else tuple(self._handle.strides)

    @property
    def shape(self) -> tuple[size_t, ...]:
        r"""
        Retrieve the shape of the allocation.

        :returns: The shape of the allocation.
        :rtype: tuple[int, ...]
        """
        if self._store.ndim == 0:
            return ()

        if self._shape is not None:
            return self._shape

        cdef _Domain domain = self._store._handle.domain()
        cdef _DomainPoint lo = domain.lo()
        cdef _DomainPoint hi = domain.hi()
        cdef int32_t ndim = domain.get_dim()

        self._shape = tuple(max(hi[i] - lo[i] + 1, 0) for i in range(ndim))
        return self._shape

    @property
    def target(self) -> StoreTarget:
        r"""
        Return the type of memory held by this ``InlineAllocation``.

        :returns: The memory type.
        :rtype: StoreTarget
        """
        cdef StoreTarget ret

        with nogil:
            ret = self._handle.target

        return ret

    cdef dict _get_array_interface(self):
        cdef Type ty = self._store.type
        cdef tuple shape = self.shape

        if math.prod(shape) == 0:
            # For some reason NumPy doesn't like a null pointer even when the
            # array size is 0, so we just make an empty ndarray and return its
            # array interface object
            return np.empty(
                shape, dtype=ty.to_numpy_dtype()
            ).__array_interface__

        return {
            "version": 3,
            "shape": shape,
            "typestr": ty.to_numpy_dtype().str,
            "data": (self.ptr, False),
            "strides": self.strides,
        }

    @property
    def __array_interface__(self) -> dict[str, Any]:
        r"""
        Retrieve the numpy-compatible array representation of the allocation.

        :returns: The numpy array interface dict.
        :rtype: dict[str, Any]

        :raises ValueError: If the allocation is allocated on the GPU.
        """
        if self.target == StoreTarget.FBMEM:
            raise ValueError(
                "Physical store in a framebuffer memory does not support "
                "the array interface"
            )
        return self._get_array_interface()

    @property
    def __cuda_array_interface__(self) -> dict[str, Any]:
        r"""
        Retrieve the cupy-compatible array representation of the allocation.

        :returns: The cupy array interface dict.
        :rtype: dict[str, Any]

        :raises ValueError: If the array is in host-only memory
        """
        if self.target not in (StoreTarget.FBMEM, StoreTarget.ZCMEM):
            raise ValueError(
                "Physical store in a host-only memory does not support "
                "the CUDA array interface"
            )
        # TODO(wonchanl): We should add a Legate-managed stream to the returned
        # interface object
        return self._get_array_interface()

    def __getbuffer__(self, Py_buffer *buffer, int flags) -> None:
        # np.asarray() seemingly stores a reference to whatever it is called
        # on, so this should avoid getting us GC'ed before the array is. The
        # filled-in buffer object will separately store a reference to the
        # numpy array, so we are transitively kept alive.
        self_np = np.asarray(_OnlyArrayInterface(self))
        PyObject_GetBuffer(self_np, buffer, flags)

    def __str__(self) -> str:
        r"""
        Return a human-readable string representation of the allocation.

        Returns
        -------
        str
            The string representation.
        """
        return f"InlineAllocation({self.ptr}, {self.strides}, {self.target})"

    def __repr__(self) -> str:
        r"""
        Return a human-readable string representation of the allocation.

        Returns
        -------
        str
            The string representation.
        """
        return str(self)
