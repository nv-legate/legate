# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint32_t, uint64_t
from libcpp cimport bool

from ..mapping.mapping cimport TaskTarget

cdef class ParallelPolicy:
    def __init__(
        self,
        *,
        streaming_mode: StreamingMode = StreamingMode.OFF,
        overdecompose_factor: uint32_t = 1,
        partitioning_threshold: dict[TaskTarget, uint64_t]
        | tuple[TaskTarget, uint64_t]
        | None = None,
    ) -> None:
        """
        Parameters
        ----------
        streaming_mode: StreamingMode
            Enum that enables streaming in STRICT or RELAXED mode.
            Default = OFF
        overdecompose_factor: int
            The over-decomposition factor.
            Default = 1
        partitioning_threshold: dict[TaskTarget, uint64_t]
                                | tuple[TaskTarget, uint64_t]

            Partitioning thresholds for various processor types specified as
            either a tuple or dictionary of [TaskTarget, int]. Default thresholds
            are picked based on Legate Runtime's configuration controlled by the
            environment variable LEGATE_CONFIG.
            Default = None

        Raises
        ------
        ValueError
            If any of the following happen:
            1) overdecompose_factor < 1
            2) partitioning_threshold is not a tuple or dict
            3) partitioning_threshold is not a tuple of [TaskTarget, uint64_t]
        """
        if overdecompose_factor < 1:
            raise ValueError("overdecompose_factor must be 1 or more")
        self._handle = _ParallelPolicy()
        self._handle.with_streaming(streaming_mode)
        self._handle.with_overdecompose_factor(<uint32_t>overdecompose_factor)

        if partitioning_threshold is not None:
            if isinstance(partitioning_threshold, dict):
                for target, threshold in partitioning_threshold.items():
                    self.set_partitioning_threshold(target, threshold)

            elif isinstance(partitioning_threshold, tuple):
                if len(partitioning_threshold) != 2:
                    raise ValueError(
                            "partitioning_threshold must be a tuple of "
                            "(TaskTarget, int)"
                    )

                self.set_partitioning_threshold(
                        partitioning_threshold[0],
                        partitioning_threshold[1]
                )

            else:
                raise ValueError(
                    "partitioning_threshold must be either a tuple or a dict of"
                    "[TaskTarget, int]"
                )

    @property
    def streaming(self) -> bool:
        """
        :returns: True if streaming has been enabled
        :rtype: bool
        """
        return self._handle.streaming()

    @property
    def streaming_mode(self) -> StreamingMode:
        return self._handle.streaming_mode()

    @streaming_mode.setter
    def streaming_mode(self, mode: StreamingMode) -> None:
        """
        :param mode: The value to set for the streaming flag.
        :type mode: StreamingMode
        :returns: None
        :rtype: None
        """
        self._handle.with_streaming(mode)

    @property
    def overdecompose_factor(self) -> uint32_t:
        """
        :returns: The current value of the overdecompose_factor.
        :rtype: uint32_t
        """
        return self._handle.overdecompose_factor()

    @overdecompose_factor.setter
    def overdecompose_factor(self, overdecompose_factor: uint32_t) -> None:
        """
        :param overdecompose_factor: Value to set for the overdecompose_factor.
        :type overdecompose_factor: uint32_t
        :returns: None
        :rtype: None
        """
        self._handle.with_overdecompose_factor(overdecompose_factor)

    cpdef uint64_t partitioning_threshold(self, TaskTarget target):
        """
        Get the value of partitioning_threshold for a processor type.

        Parameters
        ----------
        target: TaskTarget
            The processor type whose partitioning_threshold will be returned.

        Returns
        -------
        int
            The partitioning_threshold for the specified TaskTarget.

        """
        cdef uint64_t ret
        with nogil:
            ret = self._handle.partitioning_threshold(target)
        return ret

    cpdef void set_partitioning_threshold(
            self, target: TaskTarget, threshold: uint64_t
    ):
        """
        Set the partitioning_threshold for a processor type.

        Parameters
        ----------
        target: TaskTarget
            The processor type for which to set the partitioning_threshold.
        threshold: uint64_t
            The value to set for partitioning_threshold.
        """
        with nogil:
            self._handle.with_partitioning_threshold(target, threshold)

    @staticmethod
    cdef ParallelPolicy from_handle(_ParallelPolicy handle):
        cdef ParallelPolicy result = ParallelPolicy.__new__(ParallelPolicy)
        result._handle = handle
        return result

    def __eq__(self, object other) -> bool:
        if not isinstance(other, ParallelPolicy):
            return NotImplemented

        return self._handle == (<ParallelPolicy> other)._handle

    def __ne__(self, object other) -> bool:
        if not isinstance(other, ParallelPolicy):
            return NotImplemented

        return self._handle != (<ParallelPolicy> other)._handle
