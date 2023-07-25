# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import cython
from libcpp cimport bool
from libcpp.string cimport string


cdef extern from "core/legate_c.h" nogil:
    ctypedef enum legate_core_variant_t:
        pass

cdef extern from "core/task/task_info.h" namespace "legate" nogil:
    cdef cppclass TaskInfo:
        bool has_variant(int)
        string name()

cdef extern from "core/runtime/detail/library.h" namespace "legate::detail" nogil:
    cdef cppclass Library:
        unsigned int get_task_id(long long)
        unsigned int get_mapper_id()
        int get_reduction_op_id(long long)
        unsigned int get_projection_id(long long)
        unsigned int get_sharding_id(long long)
        const TaskInfo* find_task(long long) except+

cdef extern from "core/runtime/detail/runtime.h" namespace "legate::detail" nogil:
    cdef cppclass Runtime:
        @staticmethod
        Runtime* get_runtime()
        Library* find_library(string, bool)


cdef class CppTaskInfo:
    cdef const TaskInfo* _task_info

    @staticmethod
    cdef CppTaskInfo from_ptr(const TaskInfo* p_task_info):
        cdef CppTaskInfo result = CppTaskInfo.__new__(CppTaskInfo)
        result._task_info = p_task_info
        return result

    @property
    def valid(self) -> bool:
        return self._task_info != NULL

    @property
    def name(self) -> str:
        return self._task_info.name()

    def has_variant(self, int variant_id) -> bool:
        return self._task_info.has_variant(
            cython.cast(legate_core_variant_t, variant_id)
        )


cdef class Context:
    cdef Library* _context

    def __cinit__(self, str library_name, bool can_fail=False):
        self._context = Runtime.get_runtime().find_library(library_name.encode(), can_fail)

    def get_task_id(self, long long local_task_id) -> int:
        return self._context.get_task_id(local_task_id)

    def get_mapper_id(self) -> int:
        return self._context.get_mapper_id()

    def get_reduction_op_id(self, long long local_redop_id) -> int:
        return self._context.get_reduction_op_id(local_redop_id)

    def get_projection_id(self, long long local_proj_id) -> int:
        return self._context.get_projection_id(local_proj_id)

    def get_sharding_id(self, long long local_shard_id) -> int:
        return self._context.get_sharding_id(local_shard_id)

    def find_task(self, long long local_task_id) -> CppTaskInfo:
        try:
            return CppTaskInfo.from_ptr(self._context.find_task(local_task_id))
        except IndexError:
            return CppTaskInfo.from_ptr(NULL)
