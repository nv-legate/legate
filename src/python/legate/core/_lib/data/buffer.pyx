# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int32_t, uintptr_t
from libcpp.pair cimport pair as std_pair
from libcpp.utility cimport move as std_move

from ..geometry.point cimport _Point
from ..geometry.rect cimport Rect, _Rect, _rect_from_iterables
from ..mapping.mapping cimport StoreTarget
from ..type.types cimport Type, _Type
from .raw_allocation cimport (
    RawAllocation,
    _RawAllocation,
    compute_array_interface,
)

from operator import index as operator_index
from typing import Any


cdef _Rect _rect_from_object(object obj) except *:
    r"""
    Coerce a Python object into a ``legate::Rect``.

    Accepts either a ``Rect``, or an iterable of extents, in which case the
    domain is taken to start at the origin.
    """
    if isinstance(obj, Rect):
        return (<Rect> obj)._handle

    cdef list extents = [operator_index(extent) for extent in obj]

    return _rect_from_iterables(
        [0] * len(extents), [extent - 1 for extent in extents]
    )


cdef class Buffer(Unconstructable):
    @staticmethod
    cdef Buffer from_handle(
        const _Buffer &handle, object owner = None
    ):
        cdef Buffer result = Buffer.__new__(
            Buffer
        )

        result._handle = handle
        result._owner = owner
        return result

    @staticmethod
    def allocate(
        Type dtype, object domain, StoreTarget target
    ) -> Buffer:
        r"""
        Allocate a fresh buffer.

        The returned buffer is "local" to the task, and its lifetime is bound
        to that of the buffer object (and of any store the buffer is later
        bound to).

        Parameters
        ----------
        dtype : Type
            The type of the buffer.
        domain : Rect | Sequence[int]
            The domain of the buffer. A sequence of integers is interpreted as
            the extents of a domain starting at the origin.
        target : StoreTarget
            The kind of memory to allocate the buffer in.

        Returns
        -------
        Buffer
            The allocated buffer.
        """
        cdef _Type ty = dtype._handle
        cdef _Rect rect = _rect_from_object(domain)
        cdef _Buffer buf

        with nogil:
            buf = _Buffer.allocate(ty, rect, target)

        return Buffer.from_handle(buf)

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

    @property
    def domain(self) -> Rect:
        r"""
        :returns: The domain of the buffer.
        :rtype: Rect
        """
        cdef _Rect domain

        with nogil:
            domain = self._handle.domain()

        return Rect.from_handle(domain)

    @property
    def target(self) -> StoreTarget:
        r"""
        :returns: The kind of memory the buffer resides in.
        :rtype: StoreTarget
        """
        cdef StoreTarget target

        with nogil:
            target = self._handle.target()

        return target

    @property
    def allocation(self) -> RawAllocation:
        r"""
        :returns: The raw allocation backing the buffer.
        :rtype: RawAllocation
        """
        cdef _RawAllocation allocation

        with nogil:
            allocation = self._handle.allocation()

        return RawAllocation.from_handle(allocation)

    def reshape(self, object new_domain) -> None:
        r"""
        Reinterpret the buffer as having a different domain.

        The new domain must not exceed the extent of the buffer's underlying
        allocation.

        Parameters
        ----------
        new_domain : Rect | Sequence[int]
            The new domain of the buffer. A sequence of integers is
            interpreted as the extents of a domain starting at the origin.
        """
        cdef _Rect rect = _rect_from_object(new_domain)

        with nogil:
            self._handle.reshape(rect)

    cdef dict _array_interface(self):
        cdef Type ty = self.type
        cdef _Type handle = self._handle.type()
        cdef std_pair[void*, _Point] base_and_strides = (
            self._handle.base_and_element_strides()
        )

        return compute_array_interface(
            ty.to_numpy_dtype().str,
            handle.size(),
            self._handle.domain(),
            <uintptr_t>base_and_strides.first,
            base_and_strides.second,
        )

    @property
    def __array_interface__(self) -> dict[str, Any]:
        r"""
        Retrieve the numpy-compatible array representation of the buffer.

        :returns: The numpy array interface dict.
        :rtype: dict[str, Any]

        :raises ValueError: If the buffer is allocated on the GPU.
        """
        if self._handle.target() == StoreTarget.FBMEM:
            raise ValueError(
                "Buffer in a framebuffer memory does not support "
                "the array interface"
            )
        return self._array_interface()

    @property
    def __cuda_array_interface__(self) -> dict[str, Any]:
        r"""
        Retrieve the cupy-compatible array representation of the buffer.

        :returns: The cupy array interface dict.
        :rtype: dict[str, Any]

        :raises ValueError: If the buffer is in host-only memory.
        """
        if self._handle.target() not in (
            StoreTarget.FBMEM, StoreTarget.ZCMEM
        ):
            raise ValueError(
                "Buffer in a host-only memory does not support "
                "the CUDA array interface"
            )
        return self._array_interface()
