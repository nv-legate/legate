# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int64_t


cdef extern from "legate/runtime/resource.h" namespace "legate" nogil:
    cdef struct _ResourceConfig "legate::ResourceConfig":
        _ResourceConfig()

        int64_t max_tasks
        int64_t max_dyn_tasks
        int64_t max_reduction_ops
        int64_t max_projections
        int64_t max_shardings


cdef class ResourceConfig:
    cdef _ResourceConfig _handle
