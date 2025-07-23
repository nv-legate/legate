# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cython.cimports.cpython.ref import PyObject

from libc.stdint cimport int32_t, int64_t, uint32_t, uint64_t
from libcpp.map cimport map as std_map
from libcpp.memory cimport unique_ptr as std_unique_ptr
from libcpp cimport bool

from ..._ext.cython_libcpp.string_view cimport std_string_view
from ..data.external_allocation cimport _ExternalAllocation
from ..data.logical_array cimport LogicalArray, _LogicalArray
from ..data.logical_store cimport LogicalStore, _LogicalStore
from ..data.scalar cimport Scalar, _Scalar
from ..data.shape cimport _Shape
from ..mapping.machine cimport Machine, _Machine
from ..mapping.mapping cimport _Mapper
from ..operation.task cimport AutoTask, ManualTask, _AutoTask, _ManualTask
from ..task.exception cimport _TaskException
from ..task.variant_options cimport _VariantOptions, VariantOptions
from ..type.types cimport Type, _Type
from ..utilities.tuple cimport _tuple
from ..utilities.typedefs cimport _Domain, _LocalTaskID, VariantCode
from ..utilities.unconstructable cimport Unconstructable
from .detail.config cimport _Config
from .detail.runtime cimport _RuntimeImpl
from .library cimport Library, _Library
from .resource cimport _ResourceConfig, ResourceConfig


cdef extern from *:
    """
    #include <Python.h>
    #include <stdexcept>
    #include "legate/task/exception.h"
    #include "legate/task/detail/exception.h"

    PyObject* _LegateTaskException = nullptr;
    PyObject* _LegatePyTaskException = nullptr;

    void create_legate_task_exceptions()
    {
      _LegateTaskException =
        PyErr_NewException("legate.LegateTaskException", NULL, NULL);
      assert(_LegateTaskException);
      _LegatePyTaskException =
        PyErr_NewException("legate.LegatePyTaskException", NULL, NULL);
      assert(_LegatePyTaskException);
    }

    void handle_legate_exception() {
      try {
        // Looks like passing through any Python exceptions is a standard
        // procedure for a custom exception handler in Cython
        if (!PyErr_Occurred()) throw;

        // Re-raise a Legate task exception from C++ as a LegateTaskException
        // so we can catch it in the outer context and rewrite it to the
        // user-asked one.
      } catch (const legate::detail::PythonTaskException& exn) {
        PyObject* bytes = PyBytes_FromStringAndSize(
          reinterpret_cast<const char*>(exn.data()), exn.size()
        );

        assert(bytes);
        PyErr_SetObject(_LegatePyTaskException, bytes);
      } catch (const legate::TaskException& exn) {
        PyObject* args = nullptr;
        PyObject* message = nullptr;
        PyObject* index = nullptr;
        int result = 0;

        args = PyTuple_New(2);
        if (!args) {
          goto error;
        }
        message = PyUnicode_FromString(exn.what());
        if (!message) {
          goto error;
        }
        index = PyLong_FromLong(static_cast<long>(exn.index()));
        if (!index) {
          goto error;
        }

        result = PyTuple_SetItem(args, 0, message);
        // PyTuple_SetItem() decrements item refcnt on error, so we don't want
        // inadvertently decref it below
        message = nullptr;
        if (result) {
          goto error;
        }

        result = PyTuple_SetItem(args, 1, index);
        // PyTuple_SetItem() decrements item refcnt on error, so we don't want
        // inadvertently decref it below
        index = nullptr;
        if (result) {
          goto error;
        }

        PyErr_SetObject(_LegateTaskException, args);
        return;

      error:
        Py_XDECREF(args);
        Py_XDECREF(message);
        Py_XDECREF(index);
        assert(
          (result == 0)
          && "Error occurred setting values into Python tuple"
        );
        assert(
          false
          && "Error occurred during C++ to Python exception wrangling"
        );
      } catch (const std::bad_alloc& exn) {
        PyErr_SetString(PyExc_MemoryError, exn.what());
      } catch (const std::bad_cast& exn) {
        PyErr_SetString(PyExc_TypeError, exn.what());
      } catch (const std::domain_error& exn) {
        PyErr_SetString(PyExc_ValueError, exn.what());
      } catch (const std::invalid_argument& exn) {
        PyErr_SetString(PyExc_ValueError, exn.what());
      } catch (const std::ios_base::failure& exn) {
        PyErr_SetString(PyExc_IOError, exn.what());
      } catch (const std::out_of_range& exn) {
        PyErr_SetString(PyExc_IndexError, exn.what());
      } catch (const std::overflow_error& exn) {
        PyErr_SetString(PyExc_OverflowError, exn.what());
      } catch (const std::range_error& exn) {
        PyErr_SetString(PyExc_ArithmeticError, exn.what());
      } catch (const std::underflow_error& exn) {
        PyErr_SetString(PyExc_ArithmeticError, exn.what());
      } catch (const std::exception& exn) {
        PyErr_SetString(PyExc_RuntimeError, exn.what());
      } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Unknown exception");
      }
    }
    """
    cdef PyObject* _LegateTaskException
    cdef PyObject* _LegatePyTaskException
    cdef void create_legate_task_exceptions()
    cdef void handle_legate_exception()


cdef extern from "legate/runtime/runtime.h" namespace "legate" nogil:
    cdef cppclass _Runtime "legate::Runtime":
        _Library find_library(std_string_view) except+
        _Library find_or_create_library(
            std_string_view library_name,
            const _ResourceConfig& config,
            std_unique_ptr[_Mapper] mapper,
            const std_map[VariantCode, _VariantOptions]& default_options,
            bool* created
        ) except+
        _Library create_library(std_string_view) except+
        _AutoTask create_task(_Library, _LocalTaskID) except+
        _ManualTask create_task(
            _Library, _LocalTaskID, const _tuple[uint64_t]&
        ) except+
        _ManualTask create_task(_Library, _LocalTaskID, const _Domain&) except+
        void issue_copy(_LogicalStore, _LogicalStore) except+
        void issue_copy(_LogicalStore, _LogicalStore, int32_t) except+
        void issue_gather(_LogicalStore, _LogicalStore, _LogicalStore) except+
        void issue_gather(
            _LogicalStore, _LogicalStore, _LogicalStore, int32_t
        ) except+
        void issue_scatter(_LogicalStore, _LogicalStore, _LogicalStore) except+
        void issue_scatter(
            _LogicalStore, _LogicalStore, _LogicalStore, int32_t
        ) except+
        void issue_scatter_gather(
            _LogicalStore, _LogicalStore, _LogicalStore, _LogicalStore,
        ) except+
        void issue_scatter_gather(
            _LogicalStore, _LogicalStore, _LogicalStore, _LogicalStore, int32_t
        ) except+
        void issue_fill(_LogicalArray&, _LogicalStore) except+
        void issue_fill(_LogicalArray&, _Scalar) except+
        _LogicalStore tree_reduce(
            _Library, _LocalTaskID, _LogicalStore, int64_t
        ) except+
        void submit(_AutoTask) except +handle_legate_exception
        void submit(_ManualTask) except +handle_legate_exception
        _LogicalArray create_array(const _Type&, uint32_t, bool) except+
        _LogicalArray create_array(
            const _Shape&, const _Type&, bool, bool
        ) except+
        _LogicalArray create_array_like(const _LogicalArray&, _Type) except+
        _LogicalStore create_store(const _Type&, uint32_t) except+
        _LogicalStore create_store(const _Shape&, const _Type&, bool) except+
        _LogicalStore create_store(const _Scalar&) except+
        _LogicalStore create_store(const _Scalar&, const _Shape&) except+
        _LogicalStore create_store(
            const _Shape&, const _Type&, const void*, bool
        ) except+
        # TODO: dimension ordering should be added
        _LogicalStore create_store(
            const _Shape&, const _Type&, const _ExternalAllocation&
        ) except+
        void prefetch_bloated_instances(
            const _LogicalStore, _tuple[uint64_t], _tuple[uint64_t], bool
        ) except+
        void issue_mapping_fence() except+
        void issue_execution_fence(bool) except+
        void raise_pending_exception() except +handle_legate_exception
        uint32_t node_count() except+
        uint32_t node_id() except+
        _Machine get_machine() except+

        void start_profiling_range() except+
        void stop_profiling_range(std_string_view) except+

        _RuntimeImpl* impl() except+

        @staticmethod
        _Runtime* get_runtime() except+

    cdef void start() except+

    cdef int32_t finish() except+

    cdef bool _is_running_in_task "legate::is_running_in_task"() except+


cdef class Runtime(Unconstructable):
    cdef _Runtime* _handle

    @staticmethod
    cdef Runtime from_handle(_Runtime*)

    cdef const _Config* const config(self)

    cpdef Library find_library(self, str library_name)
    cdef tuple[Library, bool] find_or_create_library_mapper(
        self,
        str library_name,
        ResourceConfig config,
        std_unique_ptr[_Mapper] mapper,
        dict[VariantCode, VariantOptions] default_options,
    )
    cpdef tuple[Library, bool] find_or_create_library(
        self,
        str library_name,
        ResourceConfig config = *,
        dict[VariantCode, VariantOptions] default_options = *
    )
    cpdef Library create_library(self, str library_name)
    cpdef AutoTask create_auto_task(
        self, Library library, _LocalTaskID task_id)

    cpdef ManualTask create_manual_task(
        self,
        Library library,
        _LocalTaskID task_id,
        object launch_shape,
        object lower_bounds = *,
    )
    cpdef void issue_copy(
        self,
        LogicalStore target,
        LogicalStore source,
        object redop = *,
    )
    cpdef void issue_gather(
        self,
        LogicalStore target,
        LogicalStore source,
        LogicalStore source_indirect,
        object redop = *,
    )
    cpdef void issue_scatter(
        self,
        LogicalStore target,
        LogicalStore target_indirect,
        LogicalStore source,
        object redop = *,
    )
    cpdef void issue_scatter_gather(
        self,
        LogicalStore target,
        LogicalStore target_indirect,
        LogicalStore source,
        LogicalStore source_indirect,
        object redop = *,
    )
    cpdef void issue_fill(self, object array_or_store, object value)
    cpdef LogicalStore tree_reduce(
        self,
        Library library,
        _LocalTaskID task_id,
        LogicalStore store,
        int64_t radix = *,
    )
    cpdef void submit(self, object op)
    cpdef LogicalArray create_array(
        self,
        Type dtype,
        object shape = *,
        bool nullable = *,
        bool optimize_scalar = *,
        object ndim = *,
    )
    cpdef LogicalArray create_array_like(
        self, LogicalArray array, Type dtype = *
    )
    cpdef LogicalStore create_store(
        self,
        Type dtype,
        object shape = *,
        bool optimize_scalar = *,
        object ndim = *,
    )
    cpdef LogicalStore create_store_from_scalar(
        self, Scalar scalar, object shape = *
    )
    cpdef LogicalStore create_store_from_buffer(
        self, Type dtype, object shape, object data, bool read_only
    )
    cpdef void prefetch_bloated_instances(
        self,
        LogicalStore store,
        tuple low_offsets,
        tuple high_offsets,
        bool initialize = *,
    )
    cpdef void issue_mapping_fence(self)
    cpdef void issue_execution_fence(self, bool block = *)
    cpdef Machine get_machine(self)
    cpdef void finish(self)
    cpdef void add_shutdown_callback(self, object callback)
    cdef void start_profiling_range(self)
    cdef void stop_profiling_range(self, std_string_view provenance)

cdef void raise_pending_exception()
cpdef Runtime get_legate_runtime()
cpdef Machine get_machine()
cpdef bool is_running_in_task()
