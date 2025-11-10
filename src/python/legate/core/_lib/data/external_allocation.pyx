# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
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
from libcpp.memory cimport (
    make_unique as std_make_unique,
    unique_ptr as std_unique_ptr,
)
from libcpp.optional cimport make_optional as std_make_optional
from ..mapping.mapping cimport DimOrderingKind


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

    }  // namespace

    }  // namespace legate::detail
    """
    cdef _Deleter get_python_buffer_deleter(Py_buffer*)

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
