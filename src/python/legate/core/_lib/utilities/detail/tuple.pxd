# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint64_t

from ...geometry.point cimport _Point

cdef extern from "legate/utilities/tuple.h":
    cppclass _tuple "legate::tuple" [T]:
        pass
