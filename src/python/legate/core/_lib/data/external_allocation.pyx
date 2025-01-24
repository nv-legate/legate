# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


from cpython.buffer cimport (
    Py_buffer,
    PyBUF_CONTIG,
    PyBUF_CONTIG_RO,
    PyObject_GetBuffer,
)
from libcpp.memory cimport (
    make_unique as std_make_unique,
    unique_ptr as std_unique_ptr,
)
from libcpp.optional cimport make_optional as std_make_optional


cdef extern from * namespace "legate::detail":
    """
    #include "legate_defines.h"
    #include "legate/data/detail/external_allocation.h"
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
    object obj, size_t size, bool read_only
):
    cdef std_unique_ptr[Py_buffer] buffer = std_make_unique[Py_buffer]()
    cdef int return_code = PyObject_GetBuffer(
        obj, buffer.get(), PyBUF_CONTIG_RO if read_only else PyBUF_CONTIG
    )
    if return_code == -1:
        raise BufferError(
            f"{type(obj)} does not support the Python buffer protocol"
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
