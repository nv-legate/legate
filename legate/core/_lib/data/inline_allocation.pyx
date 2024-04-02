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


import math

import numpy as np

from libc.stdint cimport int32_t, uintptr_t

from ..type.type_info cimport Type
from ..utilities.typedefs cimport Domain, DomainPoint
from .physical_store cimport PhysicalStore


cdef class InlineAllocation:
    @staticmethod
    cdef InlineAllocation create(
        PhysicalStore store, _InlineAllocation handle
    ):
        cdef InlineAllocation result = InlineAllocation.__new__(
            InlineAllocation
        )
        result._handle = handle
        result._store = store
        result._shape = None
        return result

    @property
    def ptr(self) -> uintptr_t:
        return <long>(self._handle.ptr)

    @property
    def strides(self) -> tuple[size_t, ...]:
        return () if self._store.ndim == 0 else tuple(self._handle.strides)

    @property
    def shape(self) -> tuple[size_t, ...]:
        if self._store.ndim == 0:
            return ()

        if self._shape is not None:
            return self._shape

        cdef Domain domain = self._store.domain
        cdef DomainPoint lo = domain.lo
        cdef DomainPoint hi = domain.hi
        cdef int32_t ndim = domain.dim

        self._shape = tuple(hi[i] - lo[i] + 1 for i in range(ndim))
        return self._shape

    cdef dict _get_array_interface(self):
        cdef Type ty = self._store.type
        cdef tuple shape = self.shape

        if math.prod(shape) == 0:
            # For some reason NumPy doesn't like a null pointer even when the
            # array size is 0, so we just make an empty ndarray and return its
            # array interface object
            return np.empty(
                shape, dtype=ty.to_numpy_dtype()
            ).__array_interface__

        return {
            "version": 3,
            "shape": shape,
            "typestr": ty.to_numpy_dtype().str,
            "data": (self.ptr, False),
            "strides": self.strides,
        }

    @property
    def __array_interface__(self):
        if self._store.target == StoreTarget.FBMEM:
            raise ValueError(
                "Physical store in a framebuffer memory does not support "
                "the array interface"
            )
        return self._get_array_interface()

    @property
    def __cuda_array_interface__(self):
        if self._store.target not in (StoreTarget.FBMEM, StoreTarget.ZCMEM):
            raise ValueError(
                "Physical store in a host-only memory does not support "
                "the CUDA array interface"
            )
        # TODO(wonchanl): We should add a Legate-managed stream to the returned
        # interface object
        return self._get_array_interface()

    def __str__(self) -> str:
        return f"InlineAllocation({self.ptr}, {self.strides})"

    def __repr__(self) -> str:
        return str(self)
