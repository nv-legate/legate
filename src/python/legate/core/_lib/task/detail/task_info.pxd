# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cdef extern from "legate/task/detail/task_info.h" namespace \
  "legate::detail" nogil:
    cdef cppclass _TaskInfo "legate::detail::TaskInfo":
        pass
