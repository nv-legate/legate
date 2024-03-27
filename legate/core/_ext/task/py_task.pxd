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
from __future__ import annotations

from libc.stdint cimport int64_t
from libcpp cimport bool

from ..._lib.mapping.mapping import TaskTarget

from ..._lib.operation.task cimport AutoTask
from ..._lib.partitioning.constraint cimport ConstraintProxy
from ..._lib.runtime.library cimport Library
from ..._lib.task.task_context cimport TaskContext
from .invoker cimport VariantInvoker
from .type cimport VariantList, VariantMapping

from .type import UserFunction


cdef class PyTask:
    # Cython has no support for class variables, so this must be an instance
    # variable...
    cdef readonly int64_t UNREGISTERED_ID
    cdef:
        str                         _name
        VariantInvoker              _invoker
        VariantMapping              _variants
        int64_t                     _task_id
        Library                     _library
        tuple[ConstraintProxy, ...] _constraints
        bool                        _throws

    cpdef int64_t complete_registration(self)
    cdef void _update_variant(self, func: UserFunction, variant: TaskTarget)
    cpdef void cpu_variant(self, func: UserFunction)
    cpdef void gpu_variant(self, func: UserFunction)
    cpdef void omp_variant(self, func: UserFunction)
    cdef VariantMapping _init_variants(
        self,
        func: UserFunction,
        VariantList variants,
    )
    cdef void _invoke_variant(self, TaskContext ctx, kind: TaskTarget)
    cdef void _cpu_variant(self, TaskContext ctx)
    cdef void _gpu_variant(self, TaskContext ctx)
    cdef void _omp_variant(self, TaskContext ctx)
