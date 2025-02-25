# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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

from libcpp cimport bool

from typing import Any

from ..._lib.operation.task cimport AutoTask
from ..._lib.task.task_signature cimport _TaskSignature
from ..._lib.partitioning.constraint cimport Constraint, DeferredConstraint

from .type import UserFunction

cdef class VariantInvoker:
    cdef:
        object _signature  # inspect.Signature
        tuple[str, ...] _inputs
        tuple[str, ...] _outputs
        tuple[str, ...] _reductions
        tuple[str, ...] _scalars
        tuple[DeferredConstraint, ...] _constraints
        bool _pass_task_ctx

    cdef _TaskSignature prepare_task_signature(self)

    @staticmethod
    cdef void _handle_param(
        AutoTask task,
        object expected_param,  # inspect.Parameter
        object user_param,
    )

    cdef void _prepare_params(
        self, AutoTask task, tuple[Any, ...] args, dict[str, Any] kwargs
    )

    cpdef void prepare_call(
        self,
        AutoTask task,
        tuple[Any, ...] args,
        dict[str, Any] kwargs,
        tuple[Constraint, ...] constraints = *
    )

    @staticmethod
    cdef object _get_signature(object func)  # -> inspect.Signature
    cpdef bool valid_signature(self, func: UserFunction)
    cpdef void validate_signature(self, func: UserFunction)
