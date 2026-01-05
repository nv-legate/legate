# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int64_t


cdef class ResourceConfig:
    def __init__(
        self,
        *,
        max_tasks: object = None,
        max_dyn_tasks: object = None,
        max_reduction_ops: object = None,
        max_projections: object = None,
        max_shardings: object = None
    ) -> None:
        self._handle = _ResourceConfig()

        if max_tasks is not None:
            self.max_tasks = max_tasks
        if max_dyn_tasks is not None:
            self.max_dyn_tasks = max_dyn_tasks
        if max_reduction_ops is not None:
            self.max_reduction_ops = max_reduction_ops
        if max_projections is not None:
            self.max_projections = max_projections
        if max_shardings is not None:
            self.max_shardings = max_shardings

    @property
    def max_tasks(self) -> int64_t:
        r"""
        Get the maximum number of tasks a library can register.

        :returns: The maximum number of tasks.
        :rtype: int
        """
        return self._handle.max_tasks

    @max_tasks.setter
    def max_tasks(self, int64_t max_tasks) -> None:
        r"""
        Set the maximum number of tasks a library can register.

        Parameters
        ----------
        max_tasks : int
            The maximum number of tasks.
        """
        self._handle.max_tasks = max_tasks

    @property
    def max_dyn_tasks(self) -> int64_t:
        r"""
        Get the maximum number of dynamic tasks a library can register.

        Dynamic tasks are tasks whose task ID's are dynamically generated via
        ``Library.get_new_task_id()``.

        :returns: The maximum number of dynamic tasks.
        :rtype: int
        """
        return self._handle.max_dyn_tasks

    @max_dyn_tasks.setter
    def max_dyn_tasks(self, int64_t max_dyn_tasks) -> None:
        r"""
        Set the maximum number of dynamic tasks a library can register.

        Dynamic tasks are tasks whose task ID's are dynamically generated via
        ``Library.get_new_task_id()``.

        Parameters
        ----------
        max_dyn_tasks : int
            The maximum number of dynamic tasks.
        """
        self._handle.max_dyn_tasks = max_dyn_tasks

    @property
    def max_reduction_ops(self) -> int64_t:
        r"""
        Get the maximum number of reduction operators a library can register.

        :returns: The maximum number of reduction operators.
        :rtype: int
        """
        return self._handle.max_reduction_ops

    @max_reduction_ops.setter
    def max_reduction_ops(self, int64_t max_reduction_ops) -> None:
        r"""
        Set the maximum number of reduction operators a library can register.

        When the library is created, ``max_reduction_ops`` must not exceed
        ``max_tasks``.

        Parameters
        ----------
        max_reduction_ops : int
            The maximum number of reduction operators.
        """
        self._handle.max_reduction_ops = max_reduction_ops

    @property
    def max_projections(self) -> int64_t:
        r"""
        Get the maximum number of projection operators a library can register.

        :returns: The maximum number of projection operators.
        :rtype: int
        """
        return self._handle.max_projections

    @max_projections.setter
    def max_projections(self, int64_t max_projections) -> None:
        r"""
        Set the maximum number of projection operators a library can register.

        Parameters
        ----------
        max_projections : int
            The maximum number of projection operators.
        """
        self._handle.max_projections = max_projections

    @property
    def max_shardings(self) -> int64_t:
        r"""
        Get the maximum number of sharding operators a library can register.

        :returns: The maximum number of sharding operators.
        :rtype: int
        """
        return self._handle.max_shardings

    @max_shardings.setter
    def max_shardings(self, int64_t max_shardings) -> None:
        r"""
        Set the maximum number of sharding operators a library can register.

        Parameters
        ----------
        max_shardings : int
            The maximum number of sharding operators.
        """
        self._handle.max_shardings = max_shardings
