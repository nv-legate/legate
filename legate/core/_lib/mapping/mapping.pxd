# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Note import, not cimport. We want the Python version of the enum
from ..utilities.typedefs import VariantCode


cdef extern from "core/mapping/mapping.h" namespace "legate::mapping" nogil:
    cpdef enum class TaskTarget:
        GPU
        OMP
        CPU

cdef extern from "core/mapping/mapping.h" namespace "legate::mapping" nogil:
    cpdef enum class StoreTarget:
        SYSMEM
        FBMEM
        ZCMEM
        SOCKETMEM

cdef dict[TaskTarget, VariantCode] TASK_TARGET_TO_VARIANT_KIND
