# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from libc.stdint cimport uint32_t
from libcpp cimport bool
from libcpp.string cimport string as std_string
from libcpp.vector cimport vector as std_vector

from ..data.physical_array cimport PhysicalArray, _PhysicalArray
from ..data.scalar cimport Scalar, _Scalar
from ..utilities.typedefs cimport VariantCode, _GlobalTaskID
from ..utilities.unconstructable cimport Unconstructable
from .detail.task_context cimport _TaskContextImpl


cdef extern from "core/task/task_context.h" namespace "legate" nogil:
    cdef cppclass _TaskContext "legate::TaskContext":
        _TaskContextImpl* impl() const
        _GlobalTaskID task_id() const
        VariantCode variant_kind() const

        _PhysicalArray input(uint32_t) const
        _PhysicalArray output(uint32_t) const
        _PhysicalArray reduction(uint32_t) const
        _Scalar scalar(uint32_t) const
        size_t num_inputs() const
        size_t num_outputs() const
        size_t num_reductions() const
        const std_vector[_Scalar]& scalars() except +
        size_t num_scalars() const
        bool can_raise_exception() const


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
