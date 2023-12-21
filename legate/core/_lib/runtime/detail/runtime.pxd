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
from libcpp.string cimport string

from ...mapping.detail.machine cimport _MachineImpl


cdef extern from "core/runtime/detail/machine_manager.h" namespace "legate" nogil:  # noqa E501
    cdef cppclass _MachineManagerImpl "legate::detail::MachineManager":
        void push_machine(_MachineImpl)
        void pop_machine()


cdef extern from "core/runtime/detail/provenance_manager.h" namespace "legate" nogil:  # noqa E501
    cdef cppclass _ProvenanceManagerImpl "legate::detail::ProvenanceManager":
        bool has_provenance() const
        void push_provenance(const string&)
        void pop_provenance()


cdef extern from "core/runtime/detail/runtime.h" namespace "legate" nogil:
    cdef cppclass _RuntimeImpl "legate::detail::Runtime":
        _MachineManagerImpl* machine_manager() const
        _ProvenanceManagerImpl* provenance_manager() const
