# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool
from libcpp.memory cimport unique_ptr as std_unique_ptr
from libcpp.optional cimport optional as std_optional

from ....data.physical_store cimport _PhysicalStore, PhysicalStore

from .dlpack cimport _DLManagedTensorVersioned, _DLPackVersion, _DLDevice

cdef extern from * nogil:
    ctypedef struct CUstream_st:
        pass

    ctypedef CUstream_st *CUstream "legate::CUstream"

cdef extern from "legate/utilities/detail/dlpack/to_dlpack.h" \
  namespace "legate::detail" nogil:

    std_unique_ptr[
        _DLManagedTensorVersioned, void(*)(_DLManagedTensorVersioned*)
    ] _to_dlpack "legate::detail::to_dlpack" (
            const _PhysicalStore& store,
            std_optional[bool] copy,
            std_optional[CUstream] stream,
            std_optional[_DLPackVersion] max_version,
            std_optional[_DLDevice] device
        ) except+


cpdef object to_dlpack(
    PhysicalStore store,
    object stream=*,
    object max_version=*,
    object dl_device=*,
    object copy=*
)
