# SPDX-FileCopyrightText: Copyright (c) 2026-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool

cdef extern from "legate/runtime/runtime.h" namespace "legate" nogil:

    cdef bool has_started() except+

cpdef bool runtime_has_started()
