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

from libc.stdint cimport int64_t


cdef extern from "core/runtime/resource.h" namespace "legate" nogil:
    cdef struct _ResourceConfig "legate::ResourceConfig":
        int64_t max_tasks
        int64_t max_reduction_ops
        int64_t max_projections
        int64_t max_shardings


cdef class ResourceConfig:
    cdef ResourceConfig _handle
