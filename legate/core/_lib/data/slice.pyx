# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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
