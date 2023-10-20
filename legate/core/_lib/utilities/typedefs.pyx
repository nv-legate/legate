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


from libc.stdint cimport int32_t, int64_t


cdef class DomainPoint:
    @staticmethod
    cdef DomainPoint from_handle(_DomainPoint handle):
        cdef DomainPoint result = DomainPoint.__new__(DomainPoint)
        result._handle = handle
        return result

    def __init__(self):
        self._handle = _DomainPoint()

    @property
    def dim(self) -> int32_t:
        return self._handle.get_dim()

    def __getitem__(self, int32_t idx) -> int64_t:
        return self._handle[idx]

    def __setitem__(self, int32_t idx, int64_t coord) -> None:
        self._handle[idx] = coord


cdef class Domain:
    @staticmethod
    cdef Domain from_handle(_Domain handle):
        cdef Domain result = Domain.__new__(Domain)
        result._handle = handle
        return result

    def __init__(self):
        self._handle = _Domain()

    @property
    def dim(self) -> int32_t:
        return self._handle.get_dim()

    @property
    def lo(self) -> DomainPoint:
        return DomainPoint.from_handle(self._handle.lo())

    @property
    def hi(self) -> DomainPoint:
        return DomainPoint.from_handle(self._handle.hi())
