# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cdef extern from "legate/utilities/shared_ptr.h" namespace "legate" nogil:
    cdef cppclass _SharedPtr "legate::SharedPtr" [T]:
        _SharedPtr() except+
        T* get() except+
