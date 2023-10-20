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

from libc.stdint cimport int64_t, uint32_t

from ..data.scalar cimport _Scalar
from ..type.type_info cimport _Type


cdef extern from "core/runtime/library.h" namespace "legate" nogil:
    cdef cppclass _Library "legate::Library":
        _Library()
        _Library(const _Library&)
        uint32_t get_task_id(int64_t)
        uint32_t get_mapper_id()
        uint32_t get_reduction_op_id(int64_t)
        _Scalar get_tunable(int64_t, _Type)


cdef class Library:
    cdef _Library _handle

    @staticmethod
    cdef Library from_handle(_Library)
