# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int32_t


cdef extern from "legate_defines.h" nogil:
    cdef int32_t _LEGATE_MAX_DIM "LEGATE_MAX_DIM"
