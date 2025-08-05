# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cpython.bytes cimport PyBytes_AsStringAndSize
from libc.stddef cimport size_t
from libc.stdint cimport uintptr_t
from libcpp.string cimport string as std_string

from ..data.physical_array cimport PhysicalArray
from ..data.scalar cimport Scalar
from ..utilities.typedefs cimport (
    VariantCode,
    DomainPoint_t,
    Domain_t,
    domain_point_to_py,
    domain_to_py
)
from ..utilities.unconstructable cimport Unconstructable
from ..mapping.machine cimport Machine

from ..._ext.cython_libcpp.string_view cimport (
    std_string_view,
    str_from_string_view
)

import pickle
import traceback

# Cython doesn't know about std::byte, so we have to hide the conversion from
# const char* to it.
cdef extern from *:
    """
    #include <legate/task/detail/task_context.h>
    #include <legate/task/detail/returned_python_exception.h>

    #include <cstddef>
    #include <string>

    namespace {

    void set_python_exception(
      const legate::TaskContext& ctx,
      const char *buf,
      std::size_t len,
      std::string msg)
    {
      ctx.impl()->set_exception(
        legate::detail::ReturnedPythonException{
          reinterpret_cast<const std::byte *>(buf), len, std::move(msg)
        }
      );
    }

    } // namespace
    """
    void set_python_exception(
        const _TaskContext&, const char*, size_t, std_string
    )

cdef class TaskContext(Unconstructable):
    # the defacto constructor
    @staticmethod
    cdef TaskContext from_handle(_TaskContext* ptr):
        cdef TaskContext result = TaskContext.__new__(TaskContext)
        result._handle = ptr
        result._inputs = None
        result._outputs = None
        result._reductions = None
        result._scalars = None
        return result

    cpdef _GlobalTaskID get_task_id(self):
        r"""
        Get the global task ID.

        Returns
        -------
        GlobalTaskID
            The global task ID for this task.
        """
        return self._handle.task_id()

    cpdef VariantCode get_variant_kind(self):
        r"""
        Get the ``VariantCode`` for this task.

        Returns
        -------
        VariantCode
            The variant code for this task.
        """
        return self._handle.variant_kind()

    @property
    def inputs(self) -> tuple[PhysicalArray, ...]:
        r"""
        Get the input arguments to the task.

        :returns: The input arguments to the task.
        :rtype: tuple[PhysicalArray, ...]
        """
        cdef int i

        if self._inputs is None:
            self._inputs = tuple(
                PhysicalArray.from_handle(self._handle.input(i))
                for i in range(self._handle.num_inputs())
            )
        return self._inputs

    @property
    def outputs(self) -> tuple[PhysicalArray, ...]:
        r"""
        Get the output arguments to the task.

        :returns: The output arguments to the task.
        :rtype: tuple[PhysicalArray, ...]
        """
        cdef int i

        if self._outputs is None:
            self._outputs = tuple(
                PhysicalArray.from_handle(self._handle.output(i))
                for i in range(self._handle.num_outputs())
            )
        return self._outputs

    @property
    def reductions(self) -> tuple[PhysicalArray, ...]:
        r"""
        Get the reduction arguments to the task.

        :returns: The reduction arguments to the task.
        :rtype: tuple[PhysicalArray, ...]
        """
        cdef int i

        if self._reductions is None:
            self._reductions = tuple(
                PhysicalArray.from_handle(self._handle.reduction(i))
                for i in range(self._handle.num_reductions())
            )
        return self._reductions

    @property
    def scalars(self) -> tuple[Scalar, ...]:
        r"""
        Get the scalar arguments to the task.

        :returns: The scalar arguments to the task.
        :rtype: tuple[Scalar, ...]
        """
        cdef int i

        if self._scalars is None:
            self._scalars = tuple(
                Scalar.from_handle(self._handle.scalar(i))
                for i in range(self._handle.num_scalars())
            )
        return self._scalars

    cpdef bool is_single_task(self):
        r"""
        Indicates whether only a single instance of the task is running.

        In effect, this may be used to determine if a task is parallelized.
        If this returns false, then the task is running in parallel.

        :returns: ``True`` if the task is a single task, ``False`` otherwise.
        :rtype: bool
        """
        return self._handle.is_single_task()

    @property
    def task_index(self) -> DomainPoint_t:
        r"""
        Returns the point of the task. A 0D point will be returned for a
        single task.

        :returns: The point of the task.
        :rtype: DomainPoint
        """
        return domain_point_to_py(self._handle.get_task_index())

    @property
    def launch_domain(self) -> Domain_t:
        r"""
        Returns the task group's launch domain. A single task returns an
        empty domain.

        :returns: The launch domain of the task.
        :rtype: Domain
        """
        return domain_to_py(self._handle.get_launch_domain())

    @property
    def machine(self) -> Machine:
        r"""
        Returns the current machine this task is executing on.

        :returns: The tasks machine.
        :rtype: Machine
        """
        cdef _Machine machine = self._handle.machine()

        return Machine.from_handle(machine)

    @property
    def provenance(self) -> str:
        r"""
        Returns the provenance string for the task

        :returns: The tasks provenance.
        :rtype: str
        """
        cdef std_string_view sv = self._handle.get_provenance()

        return str_from_string_view(sv)

    @property
    def task_stream(self) -> int | None:
        r"""
        Returns a pointer to the task's CUDA stream (represented as an
        integer).

        If the current task is not a GPU task, or does not have GPU support
        enabled, this method returns None.

        :returns: The CUDA stream.
        :rtype: int | None
        """
        cdef void *stream = <void*>self._handle.get_task_stream()

        if stream == NULL:
            return None
        return int(<uintptr_t>stream)

    # Not documented on purpose, this is a private function
    cpdef void set_exception(self, Exception excn) except *:
        cdef Py_ssize_t length = 0
        cdef char *buf = NULL
        cdef bytes exn_bytes = pickle.dumps(excn)
        cdef str exn_text = "".join(traceback.format_exception(excn))

        PyBytes_AsStringAndSize(exn_bytes, &buf, &length)
        set_python_exception(self._handle[0], buf, length, exn_text.encode())

    cpdef bool can_raise_exception(self):
        r"""
        Get whether a task is allowed to raise exceptions.

        If a task is not allowed to raise an exception, but still raises one,
        the runtime will print the exception and promptly abort.

        Returns
        -------
        bool
            ``True`` if this task may raise exceptions, ``False`` otherwise.
        """
        return self._handle.can_raise_exception()
