# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint8_t, uint32_t

cdef extern from "legate/partitioning/proxy.h" namespace \
  "legate::ProxyArrayArgument":
    # NOTE: intentionally not a cpdef. ArrayKind is not exposed to the user.
    cdef enum class _ProxyArrayArgumentKind \
      "legate::ProxyArrayArgument::Kind" (uint8_t):
        INPUT,
        OUTPUT,
        REDUCTION,

cdef extern from "legate/partitioning/proxy.h" namespace "legate":
    cdef cppclass _ProxyArrayArgument "legate::ProxyArrayArgument":
        pass

    cdef cppclass _ProxyInputArguments "legate::ProxyInputArguments":
        _ProxyArrayArgument operator[](uint32_t)

    cdef cppclass _ProxyOutputArguments "legate::ProxyOutputArguments":
        _ProxyArrayArgument operator[](uint32_t)

    cdef cppclass _ProxyReductionArguments "legate::ProxyReductionArguments":
        _ProxyArrayArgument operator[](uint32_t)

    cdef const _ProxyInputArguments _inputs "legate::proxy::inputs"
    cdef const _ProxyOutputArguments _outputs "legate::proxy::outputs"
    cdef const _ProxyReductionArguments _reductions "legate::proxy::reductions"

    cdef cppclass _ProxyConstraint "legate::ProxyConstraint":
        pass
