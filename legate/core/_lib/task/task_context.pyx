# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from libc.stdint cimport int64_t

from ..data.physical_array cimport PhysicalArray
from ..data.scalar cimport Scalar

import traceback


cdef class TaskContext:
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

    def __init__(self) -> None:
        raise ValueError(
            f"{type(self).__name__} objects must not be constructed directly"
        )

    cpdef int64_t get_task_id(self):
        return self._handle.task_id()

    cpdef legate_core_variant_t get_variant_kind(self):
        return self._handle.variant_kind()

    @property
    def inputs(self) -> tuple[PhysicalArray, ...]:
        if self._inputs is None:
            self._inputs = tuple(
                PhysicalArray.from_handle(self._handle.input(i))
                for i in range(self._handle.num_inputs())
            )
        return self._inputs

    @property
    def outputs(self) -> tuple[PhysicalArray, ...]:
        if self._outputs is None:
            self._outputs = tuple(
                PhysicalArray.from_handle(self._handle.output(i))
                for i in range(self._handle.num_outputs())
            )
        return self._outputs

    @property
    def reductions(self) -> tuple[PhysicalArray, ...]:
        if self._reductions is None:
            self._reductions = tuple(
                PhysicalArray.from_handle(self._handle.reduction(i))
                for i in range(self._handle.num_reductions())
            )
        return self._reductions

    @property
    def scalars(self) -> tuple[Scalar, ...]:
        if self._scalars is None:
            self._scalars = tuple(
                Scalar.from_handle(a) for a in self._handle.scalars()
            )
        return self._scalars

    cpdef void set_exception(self, Exception excn) except *:
        cdef list excn_lines = traceback.format_exception(excn)
        cdef str what = "".join(excn_lines)

        self._handle.impl().set_exception(what.encode())
