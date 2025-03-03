# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
from ..utilities.unconstructable cimport Unconstructable
from .projection cimport _SymbolicPoint


cdef extern from "legate/operation/task.h" namespace "legate" nogil:
    cdef cppclass _AutoTask "legate::AutoTask":
        _AutoTask() except+
        _AutoTask(const _AutoTask&) except+
        _Variable add_input(_LogicalArray) except+
        _Variable add_input(_LogicalArray, _Variable) except+
        _Variable add_output(_LogicalArray) except+
        _Variable add_output(_LogicalArray, _Variable) except+
        _Variable add_reduction(_LogicalArray, int32_t) except+
        _Variable add_reduction(_LogicalArray, int32_t, _Variable) except+
        void add_scalar_arg(const _Scalar& scalar) except+
        void add_constraint(_Constraint) except+
        _Variable find_or_declare_partition(_LogicalArray) except+
        _Variable declare_partition() except+
        std_string_view provenance() except+
        void set_concurrent(bool) except+
        void set_side_effect(bool) except+
        void throws_exception(bool) except+
        void add_communicator(std_string_view) except+

    cdef cppclass _ManualTask "legate::ManualTask":
        _ManualTask() except+
        _ManualTask(const _ManualTask&) except+
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
        void add_scalar_arg(const _Scalar& scalar) except+
        std_string_view provenance() except+
        void set_concurrent(bool) except+
        void set_side_effect(bool) except+
        void throws_exception(bool) except+
        void add_communicator(std_string_view) except+


cdef class AutoTask(Unconstructable):
    cdef _AutoTask _handle
    # This is not really a dict, we only use it for the keys. We want to
    # preserve 2 properties for the registered exception types:
    #
    # 1. Registering the same exception type twice should be a no-op
    # 2. The types need to be stored in the order they were registered
    #
    # So what we really want is an ordered set, but set is famously still
    # unordered in Python (as of 3.12), so what we do instead is (ab)use the
    # dict keys and set every value to None.
    cdef dict[type, None] _exception_types
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

cdef class ManualTask(Unconstructable):
    cdef _ManualTask _handle
    # See AutoTask _exception_types on why this is the way it is
    cdef dict[type, None] _exception_types

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
