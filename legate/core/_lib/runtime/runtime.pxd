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

from cython.cimports.cpython.ref import PyObject

from libc.stdint cimport int32_t, int64_t, uint32_t, uint64_t
from libcpp cimport bool

from ..._ext.cython_libcpp.string_view cimport string_view as std_string_view
from ..data.external_allocation cimport _ExternalAllocation
from ..data.logical_array cimport LogicalArray, _LogicalArray
from ..data.logical_store cimport LogicalStore, _LogicalStore
from ..data.scalar cimport Scalar, _Scalar
from ..data.shape cimport _Shape
from ..mapping.machine cimport Machine, _Machine
from ..operation.task cimport AutoTask, ManualTask, _AutoTask, _ManualTask
from ..task.exception cimport _TaskException
from ..type.type_info cimport Type, _Type
from ..utilities.tuple cimport _tuple
from ..utilities.typedefs cimport _Domain
from .detail.runtime cimport _RuntimeImpl
from .library cimport Library, _Library
from .resource cimport _ResourceConfig


cdef extern from *:
    """
    #include <Python.h>
    #include <stdexcept>
    #include "core/task/exception.h"

    PyObject* _LegateTaskException = nullptr;

    void create_legate_task_exception() {
      _LegateTaskException =
        PyErr_NewException("legate.core.LegateTaskException", NULL, NULL);
    }

    void handle_legate_exception() {
      try {
        // Looks like passing through any Python exceptions is a standard
        // procedure for a custom exception handler in Cython
        if (!PyErr_Occurred()) throw;

      // Re-raise a Legate task exception from C++ as a LegateTaskException
      // so we can catch it in the outer context and rewrite it to the
      // user-asked one.
      } catch (const legate::TaskException& exn) {
        PyObject* args = PyTuple_New(2);
        PyObject* message = PyUnicode_FromString(exn.what());
        PyObject* index = PyLong_FromLong(static_cast<long>(exn.index()));

        int result = PyTuple_SetItem(args, 0, message);
        assert(result == 0);
        result = PyTuple_SetItem(args, 1, index);
        assert(result == 0);
        static_cast<void>(result);

        PyErr_SetObject(_LegateTaskException, args);

      // Keep the default exception mapping
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
    cdef void create_legate_task_exception()
    cdef void handle_legate_exception()


cdef extern from "core/runtime/runtime.h" namespace "legate" nogil:
    cdef cppclass _Runtime "legate::Runtime":
        _Library find_library(std_string_view)
        _AutoTask create_task(_Library, int64_t)
        _ManualTask create_task(_Library, int64_t, const _tuple[uint64_t]&)
        _ManualTask create_task(_Library, int64_t, const _Domain&)
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
            _Library, int64_t, _LogicalStore, int64_t
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
        void issue_execution_fence(bool)
        _Machine get_machine() const
        _RuntimeImpl* impl() const

        @staticmethod
        _Runtime* get_runtime()

    cdef int32_t start(int32_t, char**)

    cdef int32_t finish()

    cdef void destroy()


cdef class Runtime:
    cdef _Runtime* _handle

    @staticmethod
    cdef Runtime from_handle(_Runtime*)

    cpdef Library find_library(self, str library_name)
    cpdef AutoTask create_auto_task(self, Library library, int64_t task_id)
    cpdef ManualTask create_manual_task(
        self,
        Library library,
        int64_t task_id,
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
        int64_t task_id,
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
    cpdef LogicalArray create_array_like(self, LogicalArray array, Type dtype)
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
    cpdef void issue_execution_fence(self, bool block = *)
    cpdef Machine get_machine(self)
    cpdef void destroy(self)
    cpdef void add_shutdown_callback(self, object callback)

cpdef Runtime get_legate_runtime()
cpdef Machine get_machine()
