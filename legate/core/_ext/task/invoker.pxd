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

from libcpp cimport bool

from typing import Any

from cython.dataclasses cimport dataclass

from ..._lib.operation.task cimport AutoTask
from ..._lib.partitioning.constraint cimport ConstraintProxy
from .type cimport ConstraintSet

from .type import UserFunction


@dataclass
cdef class SeenObjTuple:  # private!
    cdef:
        bool   seen
        object value

ctypedef dict[str, SeenObjTuple] ParamMapping

cdef class VariantInvoker:
    cdef:
        object          _signature  # inspect.Signature
        tuple[str, ...] _inputs
        tuple[str, ...] _outputs
        tuple[str, ...] _reductions
        tuple[str, ...] _scalars

    @staticmethod
    cdef void _handle_param(
        AutoTask task,
        ParamMapping handled,
        object expected_param,  # inspect.Parameter
        object user_param,
    )

    cdef ParamMapping _prepare_params(
        self, AutoTask task, tuple[Any, ...] args, dict[str, Any] kwargs
    )

    @staticmethod
    cdef void _prepare_constraints(
        AutoTask task,
        ParamMapping param_mapping,
        ConstraintSet constraints,
    )

    cpdef void prepare_call(
        self,
        AutoTask task,
        tuple[Any, ...] args,
        dict[str, Any] kwargs,
        ConstraintSet constraints = *
    )

    @staticmethod
    cdef object _get_signature(func: Any)  # -> inspect.Signature
    cpdef bool valid_signature(self, func: UserFunction)
    cpdef void validate_signature(self, func: UserFunction)
