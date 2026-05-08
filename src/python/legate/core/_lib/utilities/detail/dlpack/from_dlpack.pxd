# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool
from libcpp.optional cimport optional as std_optional

from ....data.external_allocation cimport _ExternalAllocation
from ....data.logical_store cimport _LogicalStore

from .dlpack cimport _DLManagedTensorVersioned, _DLManagedTensor


cdef struct _DLPackTensor:
    void* tensor
    bool is_versioned


cdef _DLPackTensor _get_dlpack_tensor(object capsule)
cdef bool _is_gpu_device(object device)
cdef _DLPackTensor _get_dlpack_tensor_from_object(
    object obj, object device, object copy
)


cdef extern from "legate/utilities/detail/dlpack/from_dlpack.h" \
  namespace "legate::detail" nogil:
    _LogicalStore _from_dlpack "legate::detail::from_dlpack" (
        _DLManagedTensorVersioned **dlm_tensor
    ) except+

    _LogicalStore _from_dlpack "legate::detail::from_dlpack" (
        _DLManagedTensor **dlm_tensor
    ) except+

    _ExternalAllocation _make_external_alloc_from_dlpack \
        "legate::detail::make_external_alloc_from_dlpack" (
            _DLManagedTensorVersioned **dlm_tensor,
            std_optional[bool] read_only
        ) except+

    _ExternalAllocation _make_external_alloc_from_dlpack \
        "legate::detail::make_external_alloc_from_dlpack" (
            _DLManagedTensor **dlm_tensor,
            std_optional[bool] read_only
        ) except+
