# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import math

import numpy as np

from libcpp.utility cimport move as std_move
from libc.stdint cimport int32_t, uintptr_t
from cpython cimport Py_buffer, PyObject_GetBuffer

from ..type.types cimport Type
from ..utilities.typedefs cimport _Domain, _DomainPoint
from ..mapping.mapping cimport StoreTarget

cdef extern from * nogil:
    r"""
    #include <legate/runtime/detail/runtime.h>

    namespace {

    void *_get_cuda_stream()
    {
      return legate::detail::Runtime::get_runtime()->get_cuda_stream();
    }

    } // namespace
    """
    void *_get_cuda_stream() except+


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


cdef dict _compute_array_interface(
    Type ty,
    tuple shape,
    tuple strides,
    uintptr_t pointer,
):
    cdef dict ret

    np_dtype = ty.to_numpy_dtype()
    if math.prod(shape) == 0:
        # For some reason NumPy doesn't like a null pointer even when the
        # array size is 0, so we just make an empty ndarray and return its
        # array interface object
        ret = np.empty(shape, dtype=np_dtype).__array_interface__
    else:
        ret = {
            "version": 3,
            "shape": shape,
            "typestr": np_dtype.str,
            "data": (pointer, False),
            "strides": strides,
        }

    cdef void *cu_stream = _get_cuda_stream()

    if cu_stream != NULL:
        # This entry is used by __cuda_array_interface__, and is ignored by
        # numpy
        ret["stream"] = int(<uintptr_t>cu_stream)

    return ret


cdef class InlineAllocation(Unconstructable):
    @staticmethod
    cdef InlineAllocation create(
        _InlineAllocation handle,
        Type ty,
        tuple shape,
        tuple strides,
        object owner
    ):
        cdef InlineAllocation result = InlineAllocation.__new__(
            InlineAllocation
        )

        result._handle = std_move(handle)
        # Store a reference to the owning object so that it isn't
        # garbage-collected before this object is
        result._owner = owner
        result._array_interface = _compute_array_interface(
            ty=ty, shape=shape, strides=strides, pointer=result.ptr
        )
        return result

    @staticmethod
    cdef tuple _compute_shape(const _Domain& domain):
        cdef _DomainPoint lo = domain.lo()
        cdef _DomainPoint hi = domain.hi()
        cdef int32_t ndim = domain.get_dim()
        cdef int32_t i = 0
        cdef tuple ret = tuple(max(hi[i] - lo[i] + 1, 0) for i in range(ndim))

        return ret

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
        return self._array_interface["strides"]

    @property
    def shape(self) -> tuple[size_t, ...]:
        r"""
        Retrieve the shape of the allocation.

        :returns: The shape of the allocation.
        :rtype: tuple[int, ...]
        """
        return self._array_interface["shape"]

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
        return self._array_interface

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
        return self._array_interface

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
