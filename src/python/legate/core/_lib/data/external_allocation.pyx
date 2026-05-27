# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from cpython.buffer cimport (
    Py_buffer,
    PyBUF_ANY_CONTIGUOUS,
    PyBUF_STRIDED,
    PyBUF_WRITABLE,
    PyBuffer_IsContiguous,
    PyObject_GetBuffer,
)
from cpython.ref cimport Py_INCREF, PyObject
from libc.stdint cimport uint32_t
from libcpp cimport bool as cpp_bool
from libcpp.memory cimport (
    make_unique as std_make_unique,
    unique_ptr as std_unique_ptr,
)
from libcpp.optional cimport (
    make_optional as std_make_optional,
    optional as std_optional,
)
from ..mapping.mapping cimport DimOrderingKind, StoreTarget
from ..utilities.detail.dlpack.dlpack cimport (
    _DLManagedTensor,
    _DLManagedTensorVersioned,
)
from ..utilities.detail.dlpack.from_dlpack cimport (
    _DLPackTensor,
    _get_dlpack_tensor_from_object,
    _make_external_alloc_from_dlpack,
)


cdef extern from * namespace "legate::detail":
    """
    #include "legate_defines.h"
    #include "legate/utilities/typedefs.h"

    namespace legate::detail {

    namespace {

    class PyBufferDeleter {
     public:
      PyBufferDeleter(Py_buffer* buffer) : buffer_{buffer}
      {
      }
      void operator()(void* ptr) noexcept
      {
        if (ptr != buffer_->buf) {
          LEGATE_ABORT(
            "The buffer being freed does not match with the Py_buffer object "
            "in the deleter"
          );
        }

        // Hold the GIL before we mutate the Python object state
        PyGILState_STATE gstate = PyGILState_Ensure();

        PyBuffer_Release(buffer_);

        PyGILState_Release(gstate);

        std::unique_ptr<Py_buffer>{buffer_}.reset();
      }

     private:
      Py_buffer* buffer_{};
    };

    legate::ExternalAllocation::Deleter get_python_buffer_deleter(
      Py_buffer* buffer)
    {
        return PyBufferDeleter{buffer};
    }

    class PySourceRefDeleter {
     public:
      PySourceRefDeleter(PyObject* source) : source_{source}
      {
      }
      void operator()(void* /*ptr*/) noexcept
      {
        // Hold the GIL before we mutate the Python object state
        PyGILState_STATE gstate = PyGILState_Ensure();

        Py_DECREF(source_);

        PyGILState_Release(gstate);
      }

     private:
      PyObject* source_{};
    };

    legate::ExternalAllocation::Deleter get_python_source_ref_deleter(
      PyObject* source)
    {
        return PySourceRefDeleter{source};
    }

    }  // namespace

    }  // namespace legate::detail
    """
    cdef _Deleter get_python_buffer_deleter(Py_buffer*)
    cdef _Deleter get_python_source_ref_deleter(PyObject*)

cdef _ExternalAllocation create_from_buffer(
    object obj, size_t size, bool read_only, DimOrderingKind order_type,
):
    cdef std_unique_ptr[Py_buffer] buffer = std_make_unique[Py_buffer]()
    cdef int flags = (0 if read_only else PyBUF_WRITABLE)

    if order_type == DimOrderingKind.CUSTOM:
        flags |= PyBUF_STRIDED
    else:
        flags |= PyBUF_ANY_CONTIGUOUS

    cdef int return_code = PyObject_GetBuffer(obj, buffer.get(), flags)
    if return_code == -1:
        raise BufferError(  # pragma: no cover
            f"{type(obj)} does not support "  # pragma: no cover
            "the Python buffer protocol."
        )

    if (order_type == DimOrderingKind.FORTRAN and
            not PyBuffer_IsContiguous(buffer.get(), "F")):
        raise BufferError(
            "Buffer expected to be Fortran order but is not F-Contiguous."
        )

    if (order_type == DimOrderingKind.C and
            not PyBuffer_IsContiguous(buffer.get(), "C")):
        raise BufferError(
            "Buffer expected to be C order but is not C-Contiguous."
        )

    if size > buffer.get().len:
        raise BufferError(
            f"Size of the buffer ({buffer.get().len}) is smaller than "
            f"the required size ({size})"
        )

    cdef void* p_buf = buffer.get().buf
    return _ExternalAllocation.create_sysmem(
        p_buf,
        size,
        read_only,
        std_make_optional[_Deleter](
            get_python_buffer_deleter(buffer.release())
        )
    )


cdef _ExternalAllocation create_from_pointer(
    size_t ptr, size_t size, bool read_only, StoreTarget target,
    uint32_t device_id = 0,
    object source = None,
):
    cdef void* c_ptr = <void*>ptr
    if c_ptr == NULL:
        raise ValueError("ptr must not be null")

    cdef std_optional[_Deleter] deleter = std_optional[_Deleter]()
    if source is not None:
        Py_INCREF(source)
        deleter = std_make_optional[_Deleter](
            get_python_source_ref_deleter(<PyObject*>source)
        )

    if target == StoreTarget.SYSMEM or target == StoreTarget.SOCKETMEM:
        return _ExternalAllocation.create_sysmem(
            c_ptr, size, read_only, deleter
        )
    elif target == StoreTarget.FBMEM:
        return _ExternalAllocation.create_fbmem(
            device_id, c_ptr, size, read_only, deleter
        )
    elif target == StoreTarget.ZCMEM:
        return _ExternalAllocation.create_zcmem(
            c_ptr, size, read_only, deleter
        )
    else:
        raise ValueError(  # pragma: no cover
            f"Unsupported store target: {target}"
        )


cdef class ExternalAllocation:

    @staticmethod
    cdef ExternalAllocation from_handle(_ExternalAllocation handle):
        cdef ExternalAllocation result = ExternalAllocation.__new__(
            ExternalAllocation
        )
        result._handle = handle
        return result

    @staticmethod
    def from_sysmem(
        size_t ptr, size_t size, bool read_only = True, object source = None,
    ) -> ExternalAllocation:
        r"""
        Wrap a CPU (system) memory pointer.

        Parameters
        ----------
        ptr : int
            Pointer to the allocation.
        size : int
            Size in bytes.
        read_only : bool
            Whether the allocation is read-only.
        source : object, optional
            Python object owning the memory; prevents GC.
        """
        return ExternalAllocation.from_handle(
            create_from_pointer(
                ptr, size, read_only, StoreTarget.SYSMEM, 0, source
            )
        )

    @staticmethod
    def from_fbmem(
        uint32_t device_id,
        size_t ptr,
        size_t size,
        bool read_only = True,
        object source = None,
    ) -> ExternalAllocation:
        r"""
        Wrap a GPU framebuffer memory pointer.

        Parameters
        ----------
        device_id : int
            Local GPU device id.
        ptr : int
            Device pointer.
        size : int
            Size in bytes.
        read_only : bool
            Whether the allocation is read-only.
        source : object, optional
            Python object owning the memory; prevents GC.
        """
        return ExternalAllocation.from_handle(
            create_from_pointer(
                ptr, size, read_only, StoreTarget.FBMEM, device_id, source
            )
        )

    @staticmethod
    def from_zcmem(
        size_t ptr, size_t size, bool read_only = True, object source = None,
    ) -> ExternalAllocation:
        r"""
        Wrap a pinned (zero-copy) memory pointer.

        Parameters
        ----------
        ptr : int
            Pointer to the allocation.
        size : int
            Size in bytes.
        read_only : bool
            Whether the allocation is read-only.
        source : object, optional
            Python object owning the memory; prevents GC.
        """
        return ExternalAllocation.from_handle(
            create_from_pointer(
                ptr, size, read_only, StoreTarget.ZCMEM, 0, source
            )
        )

    @staticmethod
    def from_dlpack(
        object x,
        /,
        *,
        object device = None,
        object copy = None,
        object read_only = None,
    ) -> ExternalAllocation:
        r"""
        Wrap a DLPack-compatible object.

        Parameters
        ----------
        x : object
            Object exposing the DLPack protocol.
        device : tuple[int, int], optional
            DLPack target device; defaults to x.__dlpack_device__().
        copy : bool, optional
            Forwarded to x.__dlpack__(copy=...).
        read_only : bool, optional
            Overrides the capsule's read-only flag when provided.
        """
        cdef _DLPackTensor dl_tensor = _get_dlpack_tensor_from_object(
            x, device, copy
        )
        cdef _DLManagedTensorVersioned * ver_ptr
        cdef _DLManagedTensor * unver_ptr
        cdef _ExternalAllocation handle
        cdef std_optional[cpp_bool] ro_opt
        if read_only is not None:
            ro_opt = std_make_optional[cpp_bool](<cpp_bool>read_only)

        with nogil:
            if dl_tensor.is_versioned:
                ver_ptr = <_DLManagedTensorVersioned *>dl_tensor.tensor
                handle = _make_external_alloc_from_dlpack(&ver_ptr, ro_opt)
            else:
                unver_ptr = <_DLManagedTensor *>dl_tensor.tensor
                handle = _make_external_alloc_from_dlpack(&unver_ptr, ro_opt)

        return ExternalAllocation.from_handle(handle)

    @property
    def read_only(self) -> bool:
        return self._handle.read_only()

    @property
    def size(self) -> int:
        return self._handle.size()
