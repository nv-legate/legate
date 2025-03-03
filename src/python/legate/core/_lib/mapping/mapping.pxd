# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cdef extern from "legate/mapping/mapping.h" namespace "legate::mapping" nogil:
    cpdef enum class TaskTarget:
        GPU
        OMP
        CPU

cdef extern from "legate/mapping/mapping.h" namespace "legate::mapping" nogil:
    cpdef enum class StoreTarget:
        SYSMEM
        FBMEM
        ZCMEM
        SOCKETMEM
