# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool
from libcpp.vector cimport vector as std_vector
from libcpp.optional cimport optional as std_optional

from ..._ext.cython_libcpp.string_view cimport std_string_view

cdef extern from "legate/task/variant_options.h" namespace "legate" nogil:
    cdef cppclass _VariantOptions "legate::VariantOptions":
        _VariantOptions()

        bool concurrent
        bool has_allocations
        bool elide_device_ctx_sync
        bool has_side_effect
        bool may_throw_exception
        # This declaration is a lie. Cython doesn't know about std::array.
        std_optional[std_vector[std_string_view]] communicators

        _VariantOptions& with_concurrent(bool concurrent) except+
        _VariantOptions& with_has_allocations(bool has_allocations) except+
        _VariantOptions& with_elide_device_ctx_sync(bool elide_sync) except+
        _VariantOptions& with_has_side_effect(bool side_effect) except+
        _VariantOptions& with_may_throw_exception(bool may_throw) except+
        # This declaration is a lie. It is actually an initializer_list, but
        # Cython doesn't know what those are.
        _VariantOptions& with_communicators(
            const std_vector[std_string_view] &comms
        ) except+

        bool operator==(const _VariantOptions&) except+

cdef class VariantOptions:
    cdef _VariantOptions _handle

    @staticmethod
    cdef VariantOptions from_handle(const _VariantOptions& handle)
