# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ..utilities.typedefs cimport _LocalTaskID

from .task_signature cimport _TaskSignature
from .variant_options cimport _VariantOptions

cdef extern from "legate/task/task_config.h" namespace "legate" nogil:
    cdef cppclass _TaskConfig "legate::TaskConfig":
        _TaskConfig() except+
        _TaskConfig(_LocalTaskID) except+
        _TaskConfig& with_signature(const _TaskSignature& signature) except+
        _TaskConfig& with_variant_options(
            const _VariantOptions& options
        ) except+

        _LocalTaskID task_id() except+


cdef class TaskConfig:
    cdef _TaskConfig _handle
    cdef object __weakref__
