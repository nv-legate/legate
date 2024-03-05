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
from libcpp.string cimport string


cdef extern from "core/runtime/detail/runtime.h" namespace "legate" nogil:
    cdef cppclass _RuntimeImpl "legate::detail::Runtime":
        void begin_trace(uint32_t)
        void end_trace(uint32_t)
