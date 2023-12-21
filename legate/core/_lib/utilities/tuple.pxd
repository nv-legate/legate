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

from libc.stddef cimport size_t
from libcpp.vector cimport vector as std_vector


cdef extern from "core/utilities/tuple.h" namespace "legate" nogil:
    cdef cppclass tuple[T]:
        void append_inplace(const T& value)
        const std_vector[T]& data() const
        void reserve(size_t)
