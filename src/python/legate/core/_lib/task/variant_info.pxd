# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cdef extern from "legate/task/variant_info.h" namespace "legate" nogil:
    cdef cppclass _VariantInfo "legate::VariantInfo":
        pass
