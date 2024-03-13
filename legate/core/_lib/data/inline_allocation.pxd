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
from libcpp.vector cimport vector as std_vector

from ..mapping.mapping cimport StoreTarget
from ..type.type_info cimport _Type
from ..utilities.typedefs cimport _Domain
from .physical_store cimport PhysicalStore


cdef extern from "core/data/inline_allocation.h" namespace "legate" nogil:
    cdef cppclass _InlineAllocation "legate::InlineAllocation":
        void* ptr
        std_vector[size_t] strides


cdef class InlineAllocation:
    cdef _InlineAllocation _handle
    cdef PhysicalStore _store
    cdef tuple _shape

    @staticmethod
    cdef InlineAllocation create(PhysicalStore store, _InlineAllocation handle)

    cdef dict _get_array_interface(self)
