# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from libc.stdint cimport int64_t, uintptr_t

from ..data.scalar cimport Scalar
from ..task.task_info cimport _TaskInfo
from ..type.type_info cimport Type
from ..utilities.typedefs cimport _GlobalTaskID, _LocalTaskID
from ..utilities.unconstructable cimport Unconstructable


cdef class Library(Unconstructable):
    @staticmethod
    cdef Library from_handle(_Library handle):
        cdef Library result = Library.__new__(Library)
        result._handle = handle
        return result

    cpdef _LocalTaskID get_new_task_id(self):
        r"""
        Generate a new local task ID.

        Returns
        -------
        LocalTaskID
            The new local task ID.
        """
        return self._handle.get_new_task_id()

    cpdef _GlobalTaskID get_task_id(self, _LocalTaskID local_task_id):
        r"""
        Convert a global task ID to a local task ID.

        Parameters
        ----------
        local_task_id : LocalTaskID | int
            The local task ID to convert.

        Returns
        -------
        GlobalTaskID
            The global task ID.
        """
        return self._handle.get_task_id(local_task_id)

    cpdef _GlobalRedopID get_reduction_op_id(
        self, _LocalRedopID local_redop_id
    ):
        r"""
        Convert a local reduction ID into a global reduction ID.

        Parameters
        ----------
        local_redop_id : LocalRedopID | int
            The local redop ID.

        Returns
        -------
        GlobalRedopID
            The global redop ID.
        """
        return self._handle.get_reduction_op_id(local_redop_id)

    cpdef Scalar get_tunable(self, int64_t tunable_id, Type dtype):
        r"""
        Get a tunable value.

        Parameters
        ----------
        tunable_id : int
            The ID of the tunable value to get.
        dtype : Type
            The type of the tunable value to get.

        Returns
        -------
        Scalar
            The tunable.
        """
        return Scalar.from_handle(
            self._handle.get_tunable(tunable_id, dtype._handle)
        )

    cpdef _GlobalTaskID register_task(self, TaskInfo task_info):
        r"""
        Register a task with the library.

        Parameters
        ----------
        task_info : TaskInfo
            The `TaskInfo` object describing the task.

        Returns
        -------
        GlobalTaskID
            The global task ID of the task.

        Raises
        ------
        RuntimeError
            If the `TaskInfo` object is invalid or has no variants.
        """
        cdef _LocalTaskID local_task_id = task_info.get_local_id()

        # do the check now before we potentially do things we can't undo
        task_info.validate_registered_py_variants()
        # point of no return
        self._handle.register_task(
            local_task_id,
            std_unique_ptr[_TaskInfo](task_info.release())
        )

        cdef _GlobalTaskID global_task_id = self.get_task_id(local_task_id)

        task_info.register_global_variant_callbacks(global_task_id)
        return global_task_id

    @property
    def raw_handle(self) -> uintptr_t:
        r"""
        Get the raw pointer to C++ ``Library`` object.

        :returns: The pointer to the C++ ``Library`` object.
        :rtype: int
        """
        return <uintptr_t> &self._handle
