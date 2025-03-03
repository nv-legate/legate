# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool
from libcpp.map cimport map as std_map
from libcpp.optional cimport optional as std_optional
from libcpp.string cimport string as std_string

from ..._ext.cython_libcpp.string_view cimport string_view as std_string_view
from ..utilities.typedefs cimport (
    TaskFuncPtr,
    VariantCode,
    VariantImpl,
    _GlobalTaskID,
    _LocalTaskID,
)
from ..utilities.unconstructable cimport Unconstructable
from ..utilities.shared_ptr cimport _SharedPtr
from .variant_options cimport _VariantOptions
from .variant_info cimport _VariantInfo
from .task_signature cimport _TaskSignature

from .detail.task_info cimport _TaskInfo as _DetailTaskInfo

cdef extern from "legate/task/task_info.h" namespace "legate" nogil:
    cdef cppclass _TaskInfo "legate::TaskInfo":
        _TaskInfo() except+
        _TaskInfo(std_string) except+
        std_optional[_VariantInfo] find_variant(VariantCode) except+
        std_string_view  name() except+
        # add_variant's final argument is defaulted in C++, this is the only
        # way I knew how to do the same in Cython. = {}, = (), or
        # = std_map[...]() all did not work...
        void add_variant(VariantCode, VariantImpl, TaskFuncPtr) except+
        void add_variant(
            VariantCode,
            VariantImpl,
            TaskFuncPtr,
            const std_map[VariantCode, _VariantOptions]&
        ) except +
        const _SharedPtr[_DetailTaskInfo]& impl() except+

cdef class TaskInfo(Unconstructable):
    cdef:
        _TaskInfo _handle
        _LocalTaskID _local_id
        dict _registered_variants

    @staticmethod
    cdef TaskInfo from_handle(_TaskInfo, _LocalTaskID)

    @staticmethod
    cdef TaskInfo from_variants_signature(
        _LocalTaskID local_task_id,
        str name,
        list[tuple[VariantCode, object]] variants,
        const _TaskSignature *signature
    )
    cdef void validate_registered_py_variants(self)
    cdef void register_global_variant_callbacks(self, _GlobalTaskID)
    cdef _LocalTaskID get_local_id(self)
    cpdef bool has_variant(self, VariantCode)
    cdef void add_variant_signature(
        self,
        VariantCode variant_kind,
        object fn,
        const _TaskSignature *signature
    )
    cpdef void add_variant(self, VariantCode, object)
