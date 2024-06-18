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

from libc.stdint cimport int32_t, uint32_t
from libcpp cimport bool

from ..type.type_info cimport _Type
from ..utilities.typedefs cimport Domain, _Domain
from ..utilities.unconstructable cimport Unconstructable
from .physical_store cimport PhysicalStore, _PhysicalStore


cdef extern from "core/data/physical_array.h" namespace "legate" nogil:
    cdef cppclass _PhysicalArray "legate::PhysicalArray":
        bool nullable()
        int32_t dim()
        _Type type()
        bool nested()
        _PhysicalStore data() except+
        _PhysicalStore null_mask() except+
        _PhysicalArray child(uint32_t index) except+
        _Domain domain() except+

cdef class PhysicalArray(Unconstructable):
    cdef _PhysicalArray _handle

    @staticmethod
    cdef PhysicalArray from_handle(const _PhysicalArray &array)

    cpdef PhysicalStore data(self)

    cpdef PhysicalStore null_mask(self)

    cpdef PhysicalArray child(self, uint32_t index)

    cpdef Domain domain(self)
