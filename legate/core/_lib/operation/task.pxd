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
from libcpp cimport bool
from libcpp.optional cimport optional as std_optional
from libcpp.string cimport string as std_string

from ..._ext.cython_libcpp.string_view cimport string_view as std_string_view
from ..data.logical_array cimport LogicalArray, _LogicalArray
from ..data.logical_store cimport _LogicalStore, _LogicalStorePartition
from ..data.scalar cimport _Scalar
from ..partitioning.constraint cimport (
    Constraint,
    Variable,
    _Constraint,
    _Variable,
)
from ..utilities.tuple cimport _tuple
from .projection cimport _SymbolicPoint


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
        void add_communicator(std_string_view)

    cdef cppclass _ManualTask "legate::ManualTask":
        _ManualTask()
        _ManualTask(const _ManualTask&)
        void add_input(_LogicalStore) except+
        void add_input(
            _LogicalStorePartition, std_optional[_SymbolicPoint]
        ) except+
        void add_output(_LogicalStore) except+
        void add_output(
            _LogicalStorePartition, std_optional[_SymbolicPoint]
        ) except+
        void add_reduction(_LogicalStore, int32_t) except+
        void add_reduction(
            _LogicalStorePartition, int32_t, std_optional[_SymbolicPoint]
        ) except+
        void add_scalar_arg(const _Scalar& scalar)
        const std_string provenance() const
        void set_concurrent(bool)
        void set_side_effect(bool)
        void throws_exception(bool)
        void add_communicator(std_string_view)


cdef class AutoTask:
    cdef _AutoTask _handle
    cdef list[type] _exception_types
    cdef bool _locked

    @staticmethod
    cdef AutoTask from_handle(_AutoTask)

    cpdef void lock(self)
    cpdef Variable add_input(
        self, object array_or_store, object partition = *
    )
    cpdef Variable add_output(
        self, object array_or_store, object partition = *
    )
    cpdef Variable add_reduction(
        self, object array_or_store, int32_t redop, object partition = *
    )
    cpdef void add_scalar_arg(self, object value, object dtype = *)
    cpdef void add_constraint(self, Constraint constraint)
    cpdef Variable find_or_declare_partition(self, LogicalArray array)
    cpdef Variable declare_partition(self)
    cpdef str provenance(self)
    cpdef void set_concurrent(self, bool concurrent)
    cpdef void set_side_effect(self, bool has_side_effect)
    cpdef void throws_exception(self, type exception_type)
    cpdef void add_communicator(self, str name)
    cpdef void execute(self)
    cpdef void add_alignment(
        self, object array_or_store1, object array_or_store2
    )
    cpdef void add_broadcast(self, object array_or_store, object axes = *)
    cpdef void add_nccl_communicator(self)
    cpdef void add_cpu_communicator(self)
    cpdef void add_cal_communicator(self)

cdef class ManualTask:
    cdef _ManualTask _handle
    cdef list[type] _exception_types

    @staticmethod
    cdef ManualTask from_handle(_ManualTask)

    cpdef void add_input(self, object arg, object projection = *)
    cpdef void add_output(self, object arg, object projection = *)
    cpdef void add_reduction(
        self, object arg, int32_t redop, object projection = *
    )
    cpdef void add_scalar_arg(self, object value, object dtype = *)
    cpdef str provenance(self)
    cpdef void set_concurrent(self, bool concurrent)
    cpdef void set_side_effect(self, bool has_side_effect)
    cpdef void throws_exception(self, type exception_type)
    cpdef void add_communicator(self, str name)
    cpdef void execute(self)
    cpdef void add_nccl_communicator(self)
    cpdef void add_cpu_communicator(self)
    cpdef void add_cal_communicator(self)
