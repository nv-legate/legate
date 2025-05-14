# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int32_t
from libcpp.optional cimport optional as std_optional
from libcpp.string cimport string as std_string

from ..._ext.cython_libcpp.string_view cimport std_string_view
from ..mapping.machine cimport Machine, _Machine
from ..runtime.exception_mode cimport ExceptionMode
from .parallel_policy cimport ParallelPolicy, _ParallelPolicy


cdef extern from "legate/tuning/scope.h" namespace "legate" nogil:
    cdef cppclass _Scope "legate::Scope":
        _Scope() except+

        void set_priority(int32_t) except+
        void set_exception_mode(ExceptionMode) except+
        void set_provenance(std_string) except+
        void set_machine(_Machine) except+
        void set_parallel_policy(_ParallelPolicy) except+

        @staticmethod
        int32_t priority() except+

        @staticmethod
        ExceptionMode exception_mode() except+

        @staticmethod
        std_string_view provenance() except+

        @staticmethod
        _Machine machine() except+

        @staticmethod
        _ParallelPolicy parallel_policy() except+


cdef class Scope:
    cdef:
        int32_t _priority
        ExceptionMode _exception_mode
        str _provenance
        Machine _machine
        ParallelPolicy _parallel_policy
        std_optional[_Scope] _handle
