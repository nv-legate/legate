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


cdef class ResourceConfig:
    @property
    def max_tasks(self) -> int64_t:
        return self._handle.max_tasks

    @max_tasks.setter
    def max_tasks(self, int64_t max_tasks):
        self._handle.max_tasks = max_tasks

    @property
    def max_reduction_ops(self) -> int64_t:
        return self._handle.max_reduction_ops

    @max_reduction_ops.setter
    def max_reduction_ops(self, int64_t max_reduction_ops):
        self._handle.max_reduction_ops = max_reduction_ops
