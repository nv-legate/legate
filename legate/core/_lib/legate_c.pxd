# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from libc.stdint cimport int32_t, uint16_t


cdef extern from "core/legate_c.h" nogil:
    cdef cppclass __half:
        __half()
        __half(float)
        uint16_t raw() const

    float __convert_halfint_to_float(uint16_t)

    ctypedef enum legate_core_type_code_t:
        _NULL "NULL_LT"
        _BOOL "BOOL_LT"
        _INT8 "INT8_LT"
        _INT16 "INT16_LT"
        _INT32 "INT32_LT"
        _INT64 "INT64_LT"
        _UINT8 "UINT8_LT"
        _UINT16 "UINT16_LT"
        _UINT32 "UINT32_LT"
        _UINT64 "UINT64_LT"
        _FLOAT16 "FLOAT16_LT"
        _FLOAT32 "FLOAT32_LT"
        _FLOAT64 "FLOAT64_LT"
        _COMPLEX64 "COMPLEX64_LT"
        _COMPLEX128 "COMPLEX128_LT"
        _BINARY "BINARY_LT"
        _FIXED_ARRAY "FIXED_ARRAY_LT"
        _STRUCT "STRUCT_LT"
        _STRING "STRING_LT"

    ctypedef enum legate_core_reduction_op_kind_t:
        _ADD "ADD_LT"
        _SUB "SUB_LT"
        _MUL "MUL_LT"
        _DIV "DIV_LT"
        _MAX "MAX_LT"
        _MIN "MIN_LT"
        _OR  "OR_LT"
        _AND "AND_LT"
        _XOR "XOR_LT"

    ctypedef enum legate_core_task_priority_t:
        _LEGATE_CORE_DEFAULT_TASK_PRIORITY "LEGATE_CORE_DEFAULT_TASK_PRIORITY"

    cpdef enum legate_core_variant_t:
        _LEGATE_NO_VARIANT "LEGATE_NO_VARIANT"
        _LEGATE_CPU_VARIANT "LEGATE_CPU_VARIANT"
        _LEGATE_GPU_VARIANT "LEGATE_GPU_VARIANT"
        _LEGATE_OMP_VARIANT "LEGATE_OMP_VARIANT"

cdef extern from "legate_defines.h" nogil:
    cdef int32_t _LEGATE_MAX_DIM "LEGATE_MAX_DIM"
