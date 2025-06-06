# SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cdef extern from "legate/utilities/abort.h" nogil:
    void LEGATE_ABORT(...)
