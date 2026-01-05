# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ...utilities.unconstructable cimport Unconstructable
from .config cimport _Config


cdef class Config(Unconstructable):
    """
    Configuration access interface.

    .. warning::
        This class is considered an implementation detail and its members
        have no guarantee of stability across versions.
    """

    @staticmethod
    cdef Config from_handle(const _Config* handle):
        cdef Config result = Config.__new__(Config)
        result._handle = handle
        return result

    @property
    def profile(self) -> bool:
        r"""
        Whether profiling is enabled.

        :returns: True if profiling is enabled, False otherwise.
        :rtype: bool
        """
        return self._handle.profile()

    @property
    def provenance(self) -> bool:
        r"""
        Whether provenance tracking is enabled.

        :returns: True if provenance tracking is enabled, False otherwise.
        :rtype: bool
        """
        return self._handle.provenance()
