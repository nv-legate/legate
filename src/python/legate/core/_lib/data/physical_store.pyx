# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp.utility cimport move as std_move
from libc.stdint cimport int32_t

from ..type.types cimport Type
from ..utilities.typedefs cimport (
    _DomainPoint,
    domain_point_from_iterable,
    domain_to_py,
    Domain_t
)
from ..utilities.unconstructable cimport Unconstructable
from .inline_allocation cimport _InlineAllocation, InlineAllocation
from .buffer cimport _TaskLocalBuffer, TaskLocalBuffer

from typing import Any

cdef class PhysicalStore(Unconstructable):
    @staticmethod
    cdef PhysicalStore from_handle(_PhysicalStore handle):
        cdef PhysicalStore result = PhysicalStore.__new__(PhysicalStore)
        result._handle = handle
        return result

    @property
    def ndim(self) -> int32_t:
        r"""
        Get the number of dimensions in the store

        :returns: The number of dimensions in the store.
        :rtype: int
        """
        return self._handle.dim()

    @property
    def type(self) -> Type:
        r"""
        Get the type of the store.

        :returns: The type of the store.
        :rtype: Type
        """
        return Type.from_handle(self._handle.type())

    @property
    def domain(self) -> Domain_t:
        r"""
        Get the `Domain` of the store.

        :returns: The domain of the store.
        :rtype: Domain
        """
        return domain_to_py(self._handle.domain())

    @property
    def target(self) -> StoreTarget:
        r"""
        Get the kind of memory in which this store resides.

        :returns: The memory kind.
        :rtype: StoreTarget
        """
        return self._handle.target()

    cpdef TaskLocalBuffer create_output_buffer(
        self,
        object shape,
        bool bind = True
    ):
        r"""
        Create a buffer for the store to potentially bind to later on. The
        store must be unbound. The created buffer will have the same type as
        the store itself.

        Passing ``bind=False`` may be useful when creating multiple temporary
        buffers and choosing later which one to bind to the store. Buffers
        created this way can be bound to any unbound store in the task. In
        other words, a store doesn't need to create the buffer that it later
        binds to, the buffer can be created by some other store.

        The returned buffer is "local" to the task, and its lifetime is bound
        to that of the task. When the task ends, the buffer is also destroyed.

        For this reason, the returned object should only be used for temporary
        scratch-space inside a task body or for later binding to a store.

        Parameters
        ----------
        shape : Sequence[int]
            The shape of the buffer to create.
        bind : bool, True
            Whether to bind the created buffer to the store immediately.

        Returns
        -------
        TaskLocalBuffer
            The created buffer.
        """
        cdef _DomainPoint dp = domain_point_from_iterable(shape)
        cdef _TaskLocalBuffer buf

        with nogil:
            buf = self._handle.create_output_buffer(dp, bind)

        # Attach a reference to ourselves so we aren't garbage collected before
        # the buffer is
        return TaskLocalBuffer.from_handle(buf, owner=self)

    cpdef void bind_data(self, TaskLocalBuffer buffer, object extent = None):
        r"""
        Binds a buffer to the store.

        Valid only when the store is unbound and has not yet been bound to
        another buffer.

        ``extent`` may be smaller than the actual extent of ``buffer``. If not
        given, the extent of ``buffer`` is used instead.

        Parameters
        ----------
        buffer : TaskLocalBuffer
            The buffer to bind.
        extent : Sequence[int], optional
            Extent of the buffer.
        """
        cdef _DomainPoint cpp_extent

        if extent is None:
            cpp_extent = (
                buffer._handle.domain().hi() - buffer._handle.domain().lo() + 1
            )
        else:
            cpp_extent = domain_point_from_iterable(extent)

        try:
            with nogil:
                self._handle.bind_data(buffer._handle, cpp_extent)
        except ValueError as ve:
            # C++ throws invalid_argument when the types don't match (which
            # Cython translates to ValueError). But in Python, TypeError is
            # more appropriate for this
            raise TypeError(str(ve))

    cpdef InlineAllocation get_inline_allocation(self):
        r"""
        Get the `InlineAllocation` for this store.

        Returns
        -------
        InlineAllocation
            The inline allocation object holding the raw pointer and strides.
        """
        cdef _InlineAllocation handle

        with nogil:
            handle = self._handle.get_inline_allocation()

        cdef tuple strides
        cdef tuple shape

        if self.ndim == 0:
            strides = ()
            shape = ()
        else:
            strides = tuple(handle.strides)
            shape = InlineAllocation._compute_shape(self._handle.domain())

        return InlineAllocation.create(
            handle=std_move(handle),
            ty=self.type,
            shape=shape,
            strides=strides,
            owner=self,
        )

    @property
    def __array_interface__(self) -> dict[str, Any]:
        r"""
        Retrieve the numpy-compatible array representation of the allocation.

        Equivalent to `get_inline_allocation().__array_interface__`.

        :returns: The numpy array interface dict.
        :rtype: dict[str, Any]
        """
        return self.get_inline_allocation().__array_interface__

    @property
    def __cuda_array_interface__(self) -> dict[str, Any]:
        r"""
        Retrieve the cupy-compatible array representation of the allocation.

        Equivalent to `get_inline_allocation().__cuda_array_interface__`.

        :returns: The cupy array interface dict.
        :rtype: dict[str, Any]
        """
        return self.get_inline_allocation().__cuda_array_interface__
