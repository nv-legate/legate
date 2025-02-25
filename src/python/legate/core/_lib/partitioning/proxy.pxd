# SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from libc.stdint cimport uint8_t, uint32_t

cdef extern from "legate/partitioning/proxy.h" namespace \
  "legate::proxy::Array":
    # NOTE: intentionally not a cpdef. ArrayKind is not exposed to the user.
    cdef enum class _ArrayArgumentKind \
      "legate::proxy::ArrayArgument::Kind" (uint8_t):
        INPUT,
        OUTPUT,
        REDUCTION,

cdef extern from "legate/partitioning/proxy.h" namespace "legate::proxy":
    cdef cppclass _ArrayArgument "legate::proxy::ArrayArgument":
        pass

    cdef cppclass _InputArguments "legate::proxy::InputArguments":
        _ArrayArgument operator[](uint32_t)

    cdef cppclass _OutputArguments "legate::proxy::OutputArguments":
        _ArrayArgument operator[](uint32_t)

    cdef cppclass _ReductionArguments "legate::proxy::ReductionArguments":
        _ArrayArgument operator[](uint32_t)

    cdef const _InputArguments _inputs "legate::proxy::inputs"
    cdef const _OutputArguments _outputs "legate::proxy::outputs"
    cdef const _ReductionArguments _reductions "legate::proxy::reductions"

    cdef cppclass _Constraint "legate::proxy::Constraint":
        pass
