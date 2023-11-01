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

from libc.stdint cimport int32_t, int64_t, uint32_t
from libcpp cimport bool
from libcpp.string cimport string as std_string

from ..data.logical_array cimport _LogicalArray
from ..data.logical_store cimport _LogicalStore
from ..data.scalar cimport _Scalar
from ..data.shape cimport _Shape
from ..mapping.machine cimport _Machine
from ..operation.task cimport _AutoTask, _ManualTask
from ..task.exception cimport _TaskException
from ..type.type_info cimport _Type
from .detail.runtime cimport _RuntimeImpl
from .library cimport _Library
from .resource cimport _ResourceConfig


cdef extern from "core/runtime/runtime.h" namespace "legate" nogil:
    cdef cppclass _Runtime "legate::Runtime":
        _Library find_library(std_string)
        _AutoTask create_task(_Library, int64_t)
        _ManualTask create_task(_Library, int64_t, const _Shape&)
        void issue_copy(_LogicalStore, _LogicalStore) except+
        void issue_copy(_LogicalStore, _LogicalStore, int32_t) except+
        void issue_gather(_LogicalStore, _LogicalStore, _LogicalStore) except+
        void issue_gather(
            _LogicalStore, _LogicalStore, _LogicalStore, int32_t
        ) except+
        void issue_scatter(_LogicalStore, _LogicalStore, _LogicalStore) except+
        void issue_scatter(
            _LogicalStore, _LogicalStore, _LogicalStore, int32_t
        ) except+
        void issue_scatter_gather(
            _LogicalStore, _LogicalStore, _LogicalStore, _LogicalStore,
        ) except+
        void issue_scatter_gather(
            _LogicalStore, _LogicalStore, _LogicalStore, _LogicalStore, int32_t
        ) except+
        void issue_fill(_LogicalArray&, _LogicalStore) except+
        void issue_fill(_LogicalArray&, _Scalar) except+
        _LogicalStore tree_reduce(
            _Library, int64_t, _LogicalStore, int64_t
        ) except+
        void submit(_AutoTask) except+
        void submit(_ManualTask) except+
        _LogicalArray create_array(const _Type&, uint32_t, bool) except+
        _LogicalArray create_array(
            const _Shape&, const _Type&, bool, bool
        ) except+
        _LogicalArray create_array_like(const _LogicalArray&, _Type) except+
        _LogicalStore create_store(const _Type&, uint32_t) except+
        _LogicalStore create_store(const _Shape&, const _Type&, bool) except+
        _LogicalStore create_store(const _Scalar&) except+
        _LogicalStore create_store(const _Scalar&, const _Shape&) except+
        _LogicalStore create_store(
            const _Shape&, const _Type&, const void*, bool
        ) except+
        uint32_t max_pending_exceptions() const
        void set_max_pending_exceptions(uint32_t) except+
        void raise_pending_task_exception() except+
        _TaskException check_pending_task_exception()
        void issue_execution_fence(bool)
        _Machine get_machine() const
        _RuntimeImpl* impl() const

        @staticmethod
        _Runtime* get_runtime()

    cdef int32_t start(int32_t, char**)

    cdef int32_t finish()

    cdef void destroy()


cdef class Runtime:
    cdef _Runtime* _handle

    @staticmethod
    cdef Runtime from_handle(_Runtime*)
