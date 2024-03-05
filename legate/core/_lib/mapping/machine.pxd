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

from libc.stdint cimport uint32_t
from libcpp cimport bool
from libcpp.map cimport map as std_map
from libcpp.set cimport set as std_set
from libcpp.string cimport string as std_string
from libcpp.vector cimport vector as std_vector

from ..runtime.scope cimport Scope
from ..utilities.shared_ptr cimport _SharedPtr
from .mapping cimport TaskTarget


cdef extern from "core/mapping/machine.h" namespace "legate::mapping" nogil:
    cdef cppclass _NodeRange "legate::mapping::NodeRange":
        const uint32_t low
        const uint32_t high

    cdef cppclass _ProcessorRange "legate::mapping::ProcessorRange":
        uint32_t low
        uint32_t high
        uint32_t per_node_count
        uint32_t count()
        bool empty()
        _ProcessorRange slice(uint32_t, uint32_t)
        _NodeRange get_node_range() except+
        std_string to_string()
        _ProcessorRange()
        _ProcessorRange(uint32_t, uint32_t, uint32_t)
        _ProcessorRange(const _ProcessorRange&)
        _ProcessorRange operator&(const _ProcessorRange&) except+
        bool operator==(const _ProcessorRange&)
        bool operator!=(const _ProcessorRange&)
        bool operator<(const _ProcessorRange&)

    cdef cppclass _Machine "legate::mapping::Machine":
        _Machine()
        _Machine(_Machine)
        _Machine(const _Machine&)
        _Machine(std_map[TaskTarget, _ProcessorRange] ranges)
        TaskTarget preferred_target() const
        _ProcessorRange processor_range() const
        _ProcessorRange processor_range(TaskTarget target) const
        std_vector[TaskTarget] valid_targets() const
        std_vector[TaskTarget] valid_targets_except(
            const std_set[TaskTarget]&
        ) const
        uint32_t count() const
        uint32_t count(TaskTarget) const

        std_string to_string() const
        _Machine only(TaskTarget) const
        _Machine only(const std_vector[TaskTarget]&) const
        _Machine slice(uint32_t, uint32_t, TaskTarget, bool) const
        _Machine slice(uint32_t, uint32_t, bool) const
        _Machine operator[](TaskTarget target) const
        _Machine operator[](const std_vector[TaskTarget]&) const
        bool operator==(const _Machine&) const
        bool operator!=(const _Machine&) const
        _Machine operator&(const _Machine&) except+
        bool empty() const


cdef class ProcessorRange:
    cdef _ProcessorRange _handle

    @staticmethod
    cdef ProcessorRange from_handle(_ProcessorRange)

    cpdef ProcessorRange slice(self, slice sl)
    cpdef tuple get_node_range(self)

cdef class Machine:
    cdef _Machine _handle
    cdef Scope _scope

    @staticmethod
    cdef Machine from_handle(_Machine)

    cpdef ProcessorRange get_processor_range(
        self, object target = *
    )
    cpdef tuple get_node_range(self, object target = *)
    cpdef int count(self, object target = *)
    cpdef Machine only(self, object targets)
    cpdef Machine slice(self, slice sl, object target = *)
