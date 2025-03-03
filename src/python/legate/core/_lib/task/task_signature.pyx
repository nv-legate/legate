# SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint32_t
from libcpp.optional cimport optional as std_optional
from libcpp.vector cimport vector as std_vector
from libcpp.utility cimport move as std_move

from ..partitioning.proxy cimport _ProxyConstraint


cdef _TaskSignature make_task_signature(
    uint32_t num_inputs,
    uint32_t num_outputs,
    uint32_t num_redops,
    uint32_t num_scalars,
    std_vector[_ProxyConstraint] constraints,
):
    cdef std_optional[std_vector[_ProxyConstraint]] cpp_constraints

    if not constraints.empty():
        cpp_constraints = std_move(constraints)

    cdef _TaskSignature signature

    signature.inputs(num_inputs)
    signature.outputs(num_outputs)
    signature.redops(num_redops)
    signature.scalars(num_scalars)
    signature.constraints(cpp_constraints)

    return signature
