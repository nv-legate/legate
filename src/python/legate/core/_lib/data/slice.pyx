# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int64_t
from libcpp.optional cimport optional as std_optional


cdef _Slice from_python_slice(slice sl):
    if sl.step is not None and sl.step != 1:
        raise NotImplementedError(f"Unsupported slice: {sl}")

    cdef std_optional[int64_t] start = (
        OPEN
        if sl.start is None
        else std_optional[int64_t](<int64_t> sl.start)
    )
    cdef std_optional[int64_t] stop = (
        OPEN
        if sl.stop is None
        else std_optional[int64_t](<int64_t> sl.stop)
    )
    return _Slice(start, stop)
