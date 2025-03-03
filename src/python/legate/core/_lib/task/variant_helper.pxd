# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stddef cimport size_t

from ..utilities.typedefs cimport _Processor


# note missing nogil
cdef extern from "legate/task/variant_helper.h" namespace "legate::detail":
    cdef void task_wrapper_dyn_name[T, U](
        const void *, size_t, const void *, size_t, _Processor
    )
