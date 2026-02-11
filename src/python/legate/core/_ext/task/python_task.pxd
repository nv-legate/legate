# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ..._lib.task.task_info cimport TaskInfo
from ..._lib.runtime.library cimport Library
from ..._lib.utilities.typedefs cimport _GlobalTaskID, VariantCode

cdef void register_variants(_GlobalTaskID global_task_id, dict variants)
cdef void finalize_variant_registration(
    TaskInfo task_info, Library library, VariantCode code
)
