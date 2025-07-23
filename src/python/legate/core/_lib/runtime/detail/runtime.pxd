# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint32_t
from libcpp cimport bool
from libcpp.string cimport string

from .config cimport _Config

cdef extern from "legate/runtime/detail/runtime.h" namespace "legate" nogil:
    cdef cppclass _RuntimeImpl "legate::detail::Runtime":
        void begin_trace(uint32_t)
        void end_trace(uint32_t)
        const _Config& config() except+
