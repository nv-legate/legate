# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES.
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
from ..utilities.detail.dlpack.to_dlpack cimport to_dlpack
from ..utilities.detail.dlpack.dlpack cimport DLDeviceType
from ..runtime.runtime cimport get_legate_runtime
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

    def __dlpack__(
        self,
        *,
        stream: int | Any | None = None,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[Enum, int] | None = None,
        copy = None
    ) -> object:
        r"""
        Exports the store for consumption by `from_dlpack()` as a DLPack
        capsule.

        For all of the parameters listed below, please consult the Python
        array API standard for `__dlpack__()` for further discussion on their
        meaning and semantics.

        Parameters
        ----------
        stream : int | Any | None, optional
            The stream to export the store on, if any.
        max_version: tuple[int, int] | None, optional
            The maximum DLPack version that the exported capsule should support.
        dl_device: tuple[Enum, int] | None, optional
            The device to export the store to.
        copy: bool | None, optional
            Whether to copy the underlying data or not.

        Returns
        -------
        PyCapsule
            The DLPack capsule.

        Raises
        ------
        BufferError
            If the store cannot be exported as a DLPack capsule with the given
            options.

        See Also
        --------
        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__dlpack__.html#array_api.array.__dlpack__
        """
        return to_dlpack(
            self,
            stream=stream,
            max_version=max_version,
            dl_device=dl_device,
            copy=copy
        )

    cpdef tuple[int32_t, int32_t] __dlpack_device__(self):
        r"""Returns device type and device ID in DLPack format. Meant for use
        within `from_dlpack()`.

        Returns
        -------
        tuple[int, int]
            A tuple (device_type, device_id) in DLPack format.
        """
        cdef StoreTarget target = self.target

        if target in (StoreTarget.SYSMEM, StoreTarget.SOCKETMEM):
            return (DLDeviceType.kDLCPU, 0)
        if target == StoreTarget.FBMEM:
            return (
                DLDeviceType.kDLCUDA,
                get_legate_runtime().get_current_cuda_device()
            )
        if target == StoreTarget.ZCMEM:
            # The DLPack standard says in the case of pinned memory device_id
            # should be 0 (see
            # dmlc.github.io/dlpack/latest/c_api.html#c.DLDevice.device_id).
            #
            # I.e. pinned memory is not associated with any particular
            # device. This is actually true only if the cudaHostAllocPortable
            # flag is used, but luckily that's what Realm does.
            return (DLDeviceType.kDLCUDAHost, 0)

        m = f"Unhandled store target: {target}"  # pragma: no cover
        raise AssertionError(m)  # pragma: no cover
