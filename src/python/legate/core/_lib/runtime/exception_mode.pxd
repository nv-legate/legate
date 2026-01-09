# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0


cdef extern from "legate/runtime/exception_mode.h" namespace "legate" nogil:
    cpdef enum class ExceptionMode:
        IMMEDIATE
        DEFERRED
        IGNORED
