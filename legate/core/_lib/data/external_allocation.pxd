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

from libcpp cimport bool
from libcpp.optional cimport optional as std_optional

from ..mapping.mapping cimport TaskTarget


cdef extern from "core/data/external_allocation.h" namespace "legate" nogil:
    ctypedef void (*Deleter)(void*)

    cdef cppclass _ExternalAllocation "legate::ExternalAllocation":
        _ExternalAllocation()
        bool read_only() const
        TaskTarget target() const
        void* ptr() const
        size_t size() const

        @staticmethod
        _ExternalAllocation create_sysmem(
            void*, size_t, bool, std_optional[Deleter]
        )


cdef _ExternalAllocation create_from_buffer(
    object obj, size_t size, bool read_only
)
