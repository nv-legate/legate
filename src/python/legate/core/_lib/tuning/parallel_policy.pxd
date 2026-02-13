# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint32_t, uint64_t
from libcpp cimport bool

from ..mapping.mapping cimport TaskTarget

cdef extern from "legate/tuning/parallel_policy.h" namespace "legate" nogil:
    cpdef enum class StreamingMode:
        OFF
        STRICT
        RELAXED

    cdef cppclass _ParallelPolicy "legate::ParallelPolicy":
        _ParallelPolicy() except+

        _ParallelPolicy& with_streaming(StreamingMode mode) except+
        _ParallelPolicy& with_overdecompose_factor(
                uint32_t overdecompose_factor) except+
        _ParallelPolicy& with_partitioning_threshold(
                TaskTarget target, uint64_t threshold) except+

        bool streaming() except+
        StreamingMode streaming_mode() except+
        uint32_t overdecompose_factor() except+
        uint64_t partitioning_threshold(TaskTarget target) except+

        bool operator==(const _ParallelPolicy&) except+
        bool operator!=(const _ParallelPolicy&) except+


cdef class ParallelPolicy:
    cdef _ParallelPolicy _handle

    @staticmethod
    cdef ParallelPolicy from_handle(_ParallelPolicy handle)

    cpdef uint64_t partitioning_threshold(self, TaskTarget target)
    cpdef void set_partitioning_threshold(self, TaskTarget target, uint64_t
                                          threshold)
