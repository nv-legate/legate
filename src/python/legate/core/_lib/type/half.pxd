# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cdef extern from "legate/type/half.h"  namespace "legate" nogil:
    cdef cppclass _Half "legate::Half":
        _Half()
        _Half(float)
        _Half(int)

cdef extern from * nogil:
    """
    namespace {

    [[nodiscard]] float half_to_float(legate::Half h)
    {
      return static_cast<float>(h);
    }

    } // namespace
    """
    float half_to_float(_Half)
