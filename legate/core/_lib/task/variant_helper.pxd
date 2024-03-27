# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved.  SPDX-License-Identifier:
# LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from libc.stddef cimport size_t

from ..legate_c cimport legate_core_variant_t
from ..utilities.typedefs cimport _Processor


# note missing nogil
cdef extern from "core/task/variant_helper.h" namespace "legate::detail":
    cdef void task_wrapper_dyn_name[T, U](
        const void *, size_t, const void *, size_t, _Processor
    )
