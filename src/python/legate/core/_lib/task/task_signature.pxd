# SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint32_t
from libcpp.optional cimport optional as std_optional
from libcpp.vector cimport vector as std_vector

from ..partitioning.proxy cimport _ProxyConstraint
from ..partitioning.constraint cimport ImageComputationHint

cdef extern from "legate/task/task_signature.h" namespace "legate" nogil:
    cdef cppclass _TaskSignature "legate::TaskSignature":
        _TaskSignature() except+
        _TaskSignature& inputs(uint32_t) except+
        _TaskSignature& outputs(uint32_t) except+
        _TaskSignature& scalars(uint32_t) except+
        _TaskSignature& redops(uint32_t) except+
        # This type signature are a lie. But I don't want to expose Span to
        # Cython because Cython doesn't understand templated converting
        # constructors very well. Since we will be creating these as vectors
        # anyway, it's easier to just lie to Cython.
        _TaskSignature& constraints(
            std_optional[std_vector[_ProxyConstraint]] constraints
        ) except+


# Note:
#
# There is no TaskSignature class in Python. Given the ability to introspect
# callables builtin to the language, this class is not useful. Therefore it is
# also not exposed to the user.
cdef _TaskSignature make_task_signature(
    uint32_t num_inputs,
    uint32_t num_outputs,
    uint32_t num_redops,
    uint32_t num_scalars,
    std_vector[_ProxyConstraint] constraints,
)
