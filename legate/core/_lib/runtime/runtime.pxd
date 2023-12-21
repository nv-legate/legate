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

from libc.stdint cimport int32_t, int64_t, uint32_t
from libcpp cimport bool
from libcpp.string cimport string as std_string

from ..data.external_allocation cimport _ExternalAllocation
from ..data.logical_array cimport _LogicalArray
from ..data.logical_store cimport _LogicalStore
from ..data.scalar cimport _Scalar
from ..data.shape cimport _Shape
from ..mapping.machine cimport _Machine
from ..operation.task cimport _AutoTask, _ManualTask
from ..task.exception cimport _TaskException
from ..type.type_info cimport _Type
from ..utilities.typedefs cimport _Domain
from .detail.runtime cimport _RuntimeImpl
from .library cimport _Library
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
        _Library find_library(std_string)
        _AutoTask create_task(_Library, int64_t)
        _ManualTask create_task(_Library, int64_t, const _Shape&)
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
