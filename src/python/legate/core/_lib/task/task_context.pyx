# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from cpython.bytes cimport PyBytes_AsStringAndSize

from ..data.physical_array cimport PhysicalArray
from ..data.scalar cimport Scalar
from ..utilities.typedefs cimport VariantCode
from ..utilities.unconstructable cimport Unconstructable
from .detail.returned_python_exception cimport _ReturnedPythonException

import pickle
import traceback


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
        Get the `VariantCode` for this task.

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

    # Not documented on purpose, this is a private function
    cpdef void set_exception(self, Exception excn) except *:
        cdef Py_ssize_t length = 0
        cdef char *buf = NULL
        cdef bytes exn_bytes = pickle.dumps(excn)
        cdef str exn_text = "".join(traceback.format_exception(excn))

        PyBytes_AsStringAndSize(exn_bytes, &buf, &length)
        self._handle.impl().set_exception(
            _ReturnedPythonException(buf, length, exn_text.encode())
        )

    cpdef bool can_raise_exception(self):
        r"""
        Get whether a task is allowed to raise exceptions.

        If a task for which this routine returns `False` raises an exception,
        the runtime will print the exception and promptly abort.

        Returns
        -------
        bool
            `True` if this task may raise exceptions, `False` otherwise.
        """
        return self._handle.can_raise_exception()
