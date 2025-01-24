# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
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
        r"""
        Get the maximum number of tasks a library can register.

        :returns: The maximum number of tasks.
        :rtype: int
        """
        return self._handle.max_tasks

    @max_tasks.setter
    def max_tasks(self, int64_t max_tasks):
        r"""
        Set the maximum number of tasks a library can register.

        Parameters
        ----------
        max_tasks : int
            The maximum number of tasks.
        """
        self._handle.max_tasks = max_tasks

    @property
    def max_reduction_ops(self) -> int64_t:
        r"""
        Get the maximum number of reduction operators a library can register.

        :returns: The maximum number of reduction operators.
        :rtype: int
        """
        return self._handle.max_reduction_ops

    @max_reduction_ops.setter
    def max_reduction_ops(self, int64_t max_reduction_ops):
        r"""
        Set the maximum number of reduction operators a library can register.

        When the library is created, `max_reduction_ops` must not exceed
        `max_tasks`.

        Parameters
        ----------
        max_reduction_ops : int
            The maximum number of reduction operators.
        """
        self._handle.max_reduction_ops = max_reduction_ops
