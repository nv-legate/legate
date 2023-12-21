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


cimport cython
from cpython.buffer cimport (
    Py_buffer,
    PyBUF_CONTIG,
    PyBUF_CONTIG_RO,
    PyObject_GetBuffer,
)
from libc.stdlib cimport free as std_free, malloc as std_malloc
from libcpp.optional cimport optional as std_optional


cdef extern from * namespace "legate::detail":
    """
    #include "legate_defines.h"
    #include "core/data/detail/external_allocation.h"
    #include "core/utilities/typedefs.h"

    #include <cstdlib>
    #include <optional>
    #include <unordered_map>

    namespace legate::detail {

    namespace {

    struct PyBufferInfo {
      PyBufferInfo(Py_buffer* buf) : buffer{buf} {}
      Py_buffer* buffer{};
      size_t count{1};
    };
    std::unordered_map<void*, PyBufferInfo> exported_buffers{};

    void register_python_buffer(Py_buffer* buffer) {
      const auto finder = exported_buffers.find(buffer->buf);
      if (exported_buffers.end() == finder) {
        exported_buffers.try_emplace(buffer->buf, buffer);
      } else {
        ++finder->second.count;
      }
    }

    void delete_python_buffer(void* ptr) noexcept {
      const auto finder = exported_buffers.find(ptr);
      if (exported_buffers.end() == finder) {
        log_legate().fatal(
          "Failed to find a Python object mapped to pointer %p", ptr);
        LEGATE_ABORT;
      }
      if (--finder->second.count == 0) {
        // Hold the GIL before we mutate the Python object state
        PyGILState_STATE gstate = PyGILState_Ensure();

        PyBuffer_Release(finder->second.buffer);

        PyGILState_Release(gstate);

        std::free(finder->second.buffer);
        exported_buffers.erase(finder);
      }
    }

    std::optional<ExternalAllocation::Deleter> get_python_buffer_deleter() {
        static auto deleter = std::make_optional(delete_python_buffer);
        return deleter;
    }

    }  // namespace

    }  // namespace legate::detail
    """
    cdef void register_python_buffer(Py_buffer*)
    cdef std_optional[Deleter] get_python_buffer_deleter()

cdef _ExternalAllocation create_from_buffer(
    object obj, size_t size, bool read_only
):
    cdef Py_buffer* buffer = <Py_buffer*> std_malloc(cython.sizeof(Py_buffer))
    cdef int return_code = PyObject_GetBuffer(
        obj, buffer, PyBUF_CONTIG_RO if read_only else PyBUF_CONTIG
    )
    if return_code == -1:
        std_free(buffer)
        raise BufferError(
            f"{type(obj)} does not support the Python buffer protocol"
        )

    register_python_buffer(buffer)

    return _ExternalAllocation.create_sysmem(
        buffer.buf, size, read_only, get_python_buffer_deleter()
    )
