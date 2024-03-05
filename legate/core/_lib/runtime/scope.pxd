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
from libcpp.optional cimport optional as std_optional
from libcpp.string cimport string as std_string

from ..mapping.machine cimport Machine, _Machine


cdef extern from "core/runtime/scope.h" namespace "legate" nogil:
    cdef cppclass _Scope "legate::Scope":
        _Scope()

        void set_priority(int32_t) except+
        void set_provenance(std_string) except+
        void set_machine(_Machine) except+

        @staticmethod
        int32_t priority()

        @staticmethod
        std_string provenance()

        @staticmethod
        _Machine machine()


cdef class Scope:
    cdef:
        int32_t _priority
        str _provenance
        Machine _machine
        std_optional[_Scope] _handle
