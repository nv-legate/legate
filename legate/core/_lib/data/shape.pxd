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

from libcpp.vector cimport vector as std_vector

from ..utilities.typedefs cimport _Domain


cdef extern from "core/data/shape.h" namespace "legate" nogil:
    cdef cppclass _Shape "legate::Shape":
        _Shape()
        _Shape(const std_vector[size_t]&)
        void append_inplace(const size_t& value)
        const std_vector[size_t]& data() const

    cdef _Domain to_domain(const _Shape&)

    cdef _Shape from_domain(const _Domain&)
