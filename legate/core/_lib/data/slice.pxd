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


cdef extern from "core/data/slice.h" namespace "legate" nogil:
    cdef std_optional[int64_t] OPEN "legate::Slice::OPEN"

    cdef cppclass _Slice "legate::Slice":
        _Slice()
        _Slice(std_optional[int64_t], std_optional[int64_t])


cdef _Slice from_python_slice(slice sl)
