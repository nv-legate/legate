# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from libc.stdint cimport int64_t, uint32_t
from libcpp cimport bool
from libcpp.map cimport map as std_map
from libcpp.string cimport string as std_string

from ..legate_c cimport legate_core_variant_t
from ..utilities.typedefs cimport RealmCallbackFn, VariantImpl
from .variant_options cimport _VariantOptions


cdef extern from "core/task/task_info.h" namespace "legate" nogil:
    cdef cppclass _TaskInfo "legate::TaskInfo":
        _TaskInfo(std_string)
        bool has_variant(legate_core_variant_t) const
        const std_string& name() const
        # add_variant's final argument is defaulted in C++, this is the only
        # way I knew how to do the same in Cython. = {}, = (), or
        # = std_map[...]() all did not work...
        void add_variant(
            legate_core_variant_t,
            VariantImpl,
            RealmCallbackFn,
        ) except +
        void add_variant(
            legate_core_variant_t,
            VariantImpl,
            RealmCallbackFn,
            const std_map[legate_core_variant_t, _VariantOptions]&
        ) except +

cdef class TaskInfo:
    cdef:
        _TaskInfo *_handle
        int64_t _local_id
        dict _registered_variants

    @staticmethod
    cdef TaskInfo from_handle(_TaskInfo*, int64_t)
    cdef _TaskInfo *release(self) except NULL
    cdef void validate_registered_py_variants(self)
    cdef void register_global_variant_callbacks(self, uint32_t)
    cdef int64_t get_local_id(self)
    cpdef bool has_variant(self, int)
    cpdef void add_variant(self, legate_core_variant_t, object)
