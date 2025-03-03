# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint32_t
from libcpp cimport bool
from libcpp.map cimport map as std_map
from libcpp.set cimport set as std_set
from libcpp.string cimport string as std_string
from libcpp.vector cimport vector as std_vector

from ..runtime.scope cimport Scope
from ..utilities.shared_ptr cimport _SharedPtr
from .mapping cimport TaskTarget


cdef extern from "legate/mapping/machine.h" namespace "legate::mapping" nogil:
    cdef cppclass _NodeRange "legate::mapping::NodeRange":
        const uint32_t low
        const uint32_t high

    cdef cppclass _ProcessorRange "legate::mapping::ProcessorRange":
        uint32_t low
        uint32_t high
        uint32_t per_node_count
        uint32_t count() except+
        bool empty() except+
        _ProcessorRange slice(uint32_t, uint32_t) except+
        _NodeRange get_node_range() except+
        std_string to_string() except+
        _ProcessorRange() except+
        _ProcessorRange(uint32_t, uint32_t, uint32_t) except+
        _ProcessorRange(const _ProcessorRange&) except+
        _ProcessorRange operator&(const _ProcessorRange&) except+
        bool operator==(const _ProcessorRange&) except+
        bool operator!=(const _ProcessorRange&) except+
        bool operator<(const _ProcessorRange&) except+

    cdef cppclass _Machine "legate::mapping::Machine":
        _Machine() except+
        _Machine(_Machine) except+
        _Machine(const _Machine&) except+
        _Machine(std_map[TaskTarget, _ProcessorRange] ranges) except+
        TaskTarget preferred_target() except+
        _ProcessorRange processor_range() except+
        _ProcessorRange processor_range(TaskTarget target) except+
        const std_vector[TaskTarget]& valid_targets() except+
        std_vector[TaskTarget] valid_targets_except(
            const std_set[TaskTarget]&
        ) except+
        uint32_t count() except+
        uint32_t count(TaskTarget) except+

        std_string to_string() except+
        _Machine only(TaskTarget) except+
        _Machine only(const std_vector[TaskTarget]&) except+
        _Machine slice(uint32_t, uint32_t, TaskTarget, bool) except+
        _Machine slice(uint32_t, uint32_t, bool) except+
        _Machine operator[](TaskTarget target) except+
        _Machine operator[](const std_vector[TaskTarget]&) except+
        bool operator==(const _Machine&) except+
        bool operator!=(const _Machine&) except+
        _Machine operator&(const _Machine&) except+
        bool empty() except+


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
