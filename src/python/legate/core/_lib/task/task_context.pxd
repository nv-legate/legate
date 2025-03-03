# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint32_t
from libcpp cimport bool
from libcpp.string cimport string as std_string
from libcpp.vector cimport vector as std_vector

from ..data.physical_array cimport PhysicalArray, _PhysicalArray
from ..data.scalar cimport Scalar, _Scalar
from ..utilities.typedefs cimport VariantCode, _GlobalTaskID
from ..utilities.unconstructable cimport Unconstructable
from .detail.task_context cimport _TaskContextImpl


cdef extern from "legate/task/task_context.h" namespace "legate" nogil:
    cdef cppclass _TaskContext "legate::TaskContext":
        _TaskContextImpl* impl() except+
        _GlobalTaskID task_id() except+
        VariantCode variant_kind() except+

        _PhysicalArray input(uint32_t) except+
        _PhysicalArray output(uint32_t) except+
        _PhysicalArray reduction(uint32_t) except+
        _Scalar scalar(uint32_t) except+
        size_t num_inputs() except+
        size_t num_outputs() except+
        size_t num_reductions() except+
        const std_vector[_Scalar]& scalars() except+
        size_t num_scalars() except+
        bool can_raise_exception() except+


cdef class TaskContext(Unconstructable):
    cdef:
        _TaskContext* _handle
    cdef readonly:
        tuple[PhysicalArray, ...] _inputs
        tuple[PhysicalArray, ...] _outputs
        tuple[PhysicalArray, ...] _reductions
        tuple[Scalar, ...] _scalars

    # the defacto constructor
    @staticmethod
    cdef TaskContext from_handle(_TaskContext* ptr)
    cpdef _GlobalTaskID get_task_id(self)
    cpdef VariantCode get_variant_kind(self)
    cpdef void set_exception(self, Exception) except *
    cpdef bool can_raise_exception(self)
