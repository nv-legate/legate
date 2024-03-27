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
from libc.stdint cimport uint32_t, uint64_t
from libcpp cimport bool
from libcpp.string cimport string as std_string
from libcpp.vector cimport vector as std_vector

from ..utilities.tuple cimport _tuple
from ..utilities.typedefs cimport _Domain, _DomainPoint


cdef extern from "core/data/shape.h" namespace "legate" nogil:
    cdef cppclass _Shape "legate::Shape":
        _Shape()
        _Shape(const _tuple[uint64_t]&)
        _Shape(const std_vector[uint64_t]&)

        _tuple[uint64_t] extents() except+
        size_t volume() except+
        uint32_t ndim() const

        std_string to_string()

        bool operator==(const _Shape&) const


cdef class Shape:
    cdef _Shape _handle
    cdef tuple _extents

    @staticmethod
    cdef Shape from_handle(_Shape)

    @staticmethod
    cdef _Shape from_shape_like(object)
