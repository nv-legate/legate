# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from libc.stdint cimport uint32_t

from ..type.type_info cimport Type
from .physical_store cimport PhysicalStore


cdef class PhysicalArray:
    @staticmethod
    cdef PhysicalArray from_handle(const _PhysicalArray &array):
        cdef PhysicalArray result = PhysicalArray.__new__(PhysicalArray)
        result._handle = array
        return result

    def __init__(self) -> None:
        raise ValueError(
            f"{type(self).__name__} objects must not be constructed directly"
        )

    @property
    def nullable(self) -> bool:
        return self._handle.nullable()

    @property
    def ndim(self) -> int:
        return self._handle.dim()

    @property
    def type(self) -> Type:
        return Type.from_handle(self._handle.type())

    @property
    def nested(self) -> bool:
        return self._handle.nested()

    cpdef PhysicalStore data(self):
        return PhysicalStore.from_handle(self._handle.data())

    cpdef PhysicalStore null_mask(self):
        return PhysicalStore.from_handle(self._handle.null_mask())

    cpdef PhysicalArray child(self, uint32_t index):
        return PhysicalArray.from_handle(self._handle.child(index))

    cpdef Domain domain(self):
        return Domain.from_handle(self._handle.domain())

    @property
    def __array_interface__(self) -> dict[str, Any]:
        if self.nullable or self.nested:
            raise ValueError(
                "Nested or nullable arrays don't support the array interface "
                "directly"
            )
        return self.data().__array_interface__

    @property
    def __cuda_array_interface__(self) -> dict[str, Any]:
        if self.nullable or self.nested:
            raise ValueError(
                "Nested or nullable arrays don't support the CUDA array "
                "interface directly"
            )
        return self.data().__cuda_array_interface__
