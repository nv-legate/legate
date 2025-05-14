# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint32_t
from libcpp cimport bool


cdef extern from "legate/tuning/parallel_policy.h" namespace "legate" nogil:
    cdef cppclass _ParallelPolicy "legate::ParallelPolicy":
        _ParallelPolicy() except+

        _ParallelPolicy& with_streaming(bool streaming) except+
        _ParallelPolicy& with_overdecompose_factor(
                uint32_t overdecompose_factor) except+

        bool streaming() except+
        uint32_t overdecompose_factor() except+

        bool operator==(const _ParallelPolicy&) except+
        bool operator!=(const _ParallelPolicy&) except+


cdef class ParallelPolicy:
    cdef _ParallelPolicy _handle

    @staticmethod
    cdef ParallelPolicy from_handle(_ParallelPolicy handle)
