# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool
from ...utilities.unconstructable cimport Unconstructable

cdef extern from "legate/runtime/detail/config.h" namespace "legate" nogil:
    cdef cppclass _Config "legate::detail::Config":
        bool profile() except+
        bool provenance() except+

cdef class Config(Unconstructable):
    cdef const _Config* _handle

    @staticmethod
    cdef Config from_handle(const _Config* handle)
