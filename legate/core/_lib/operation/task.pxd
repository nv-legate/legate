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

from libc.stdint cimport int32_t
from libcpp.string cimport string as std_string

from ..data.logical_array cimport _LogicalArray
from ..data.logical_store cimport _LogicalStore, _LogicalStorePartition
from ..data.scalar cimport _Scalar
from ..partitioning.constraint cimport _Constraint, _Variable


cdef extern from "core/operation/task.h" namespace "legate" nogil:
    cdef cppclass _AutoTask "legate::AutoTask":
        _AutoTask()
        _AutoTask(const _AutoTask&)
        _Variable add_input(_LogicalArray) except+
        _Variable add_input(_LogicalArray, _Variable) except+
        _Variable add_output(_LogicalArray) except+
        _Variable add_output(_LogicalArray, _Variable) except+
        _Variable add_reduction(_LogicalArray, int32_t) except+
        _Variable add_reduction(_LogicalArray, int32_t, _Variable) except+
        void add_scalar_arg(const _Scalar& scalar)
        void add_constraint(_Constraint)
        _Variable find_or_declare_partition(_LogicalArray)
        _Variable declare_partition()
        const std_string provenance() const
        void set_concurrent(bool)
        void set_side_effect(bool)
        void throws_exception(bool)
        void add_communicator(const std_string&)

    cdef cppclass _ManualTask "legate::ManualTask":
        _ManualTask()
        _ManualTask(const _ManualTask&)
        void add_input(_LogicalStore) except+
        void add_input(_LogicalStorePartition) except+
        void add_output(_LogicalStore) except+
        void add_output(_LogicalStorePartition) except+
        void add_reduction(_LogicalStore, int32_t) except+
        void add_reduction(_LogicalStorePartition, int32_t) except+
        void add_scalar_arg(const _Scalar& scalar)
        const std_string provenance() const
        void set_concurrent(bool)
        void set_side_effect(bool)
        void throws_exception(bool)
        void add_communicator(const std_string&)


cdef class AutoTask:
    cdef _AutoTask _handle

    @staticmethod
    cdef AutoTask from_handle(_AutoTask)


cdef class ManualTask:
    cdef _ManualTask _handle

    @staticmethod
    cdef ManualTask from_handle(_ManualTask)
