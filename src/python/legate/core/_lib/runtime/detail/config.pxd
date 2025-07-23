# SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool

cdef extern from "legate/runtime/detail/config.h" namespace "legate" nogil:
    cdef cppclass _Config "legate::detail::Config":
        bool profile() except+
        bool provenance() except+
