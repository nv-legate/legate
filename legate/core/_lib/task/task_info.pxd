# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from libcpp cimport bool
from libcpp.functional cimport reference_wrapper as std_reference_wrapper
from libcpp.map cimport map as std_map
from libcpp.optional cimport optional as std_optional
from libcpp.string cimport string as std_string

from ..._ext.cython_libcpp.string_view cimport string_view as std_string_view
from ..legate_c cimport legate_core_variant_t
from ..utilities.typedefs cimport (
    TaskFuncPtr,
    VariantImpl,
    _GlobalTaskID,
    _LocalTaskID,
)
from ..utilities.unconstructable cimport Unconstructable
from .variant_options cimport _VariantOptions


cdef extern from "core/task/task_info.h" namespace "legate" nogil:
    cdef cppclass _VariantInfo "legate::VariantInfo":
        pass

    cdef cppclass _TaskInfo "legate::TaskInfo":
        _TaskInfo(std_string)
        std_optional[std_reference_wrapper[const _VariantInfo]] \
            find_variant(legate_core_variant_t) const
        std_string_view  name() const
        # add_variant's final argument is defaulted in C++, this is the only
        # way I knew how to do the same in Cython. = {}, = (), or
        # = std_map[...]() all did not work...
        void add_variant(
            legate_core_variant_t,
            VariantImpl,
            TaskFuncPtr,
        ) except +
        void add_variant(
            legate_core_variant_t,
            VariantImpl,
            TaskFuncPtr,
            const std_map[legate_core_variant_t, _VariantOptions]&
        ) except +

cdef class TaskInfo(Unconstructable):
    cdef:
        _TaskInfo *_handle
        _LocalTaskID _local_id
        dict _registered_variants

    cdef void _assert_valid(self)

    @staticmethod
    cdef TaskInfo from_handle(_TaskInfo*, _LocalTaskID)
    cdef _TaskInfo *release(self) except NULL
    cdef void validate_registered_py_variants(self)
    cdef void register_global_variant_callbacks(self, _GlobalTaskID)
    cdef _LocalTaskID get_local_id(self)
    cpdef bool has_variant(self, int)
    cpdef void add_variant(self, legate_core_variant_t, object)
