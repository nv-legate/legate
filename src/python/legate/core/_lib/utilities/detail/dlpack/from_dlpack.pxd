# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ....data.logical_store cimport _LogicalStore

from .dlpack cimport _DLManagedTensorVersioned, _DLManagedTensor

cdef extern from "legate/utilities/detail/dlpack/from_dlpack.h" \
  namespace "legate::detail" nogil:
    _LogicalStore _from_dlpack "legate::detail::from_dlpack" (
        _DLManagedTensorVersioned **dlm_tensor
    ) except+

    _LogicalStore _from_dlpack "legate::detail::from_dlpack" (
        _DLManagedTensor **dlm_tensor
    ) except+
