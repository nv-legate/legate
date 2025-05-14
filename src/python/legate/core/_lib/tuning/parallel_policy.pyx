# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint32_t
from libcpp cimport bool


cdef class ParallelPolicy:
    def __init__(
        self,
        *,
        streaming: bool = False,
        overdecompose_factor: uint32_t = 1,
    ) -> None:
        """
        Parameters
        ----------
        streaming: bool
            Whether to enable streaming.
        overdecompose_factor: int
            The overdecomposition factor.
        """
        self._handle = _ParallelPolicy()
        self.streaming = streaming
        self.overdecompose_factor = overdecompose_factor

    @property
    def streaming(self) -> bool:
        """
        :returns: The current value of the streaming flag.
        :rtype: bool
        """
        return self._handle.streaming()

    @streaming.setter
    def streaming(self, streaming: bool) -> None:
        """
        :param streaming: The value to set for the streaming flag.
        :type streaming: bool
        :returns: None
        :rtype: None
        """
        self._handle.with_streaming(streaming)

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
