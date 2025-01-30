# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from libc.stdint cimport uint64_t
from libcpp.optional cimport optional as std_optional
from libcpp.vector cimport vector as std_vector

from ....._ext.cython_libcpp.string_view cimport string_view as std_string_view
from ....data.logical_array cimport LogicalArray, _LogicalArray
from ....data.shape cimport Shape, _Shape
from ....type.types cimport Type, _Type


cdef extern from "legate/experimental/io/kvikio/interface.h" \
      namespace "legate" nogil:
    # These std_string_view arguments are in reality std::filesystem::path, so
    # these prototypes are actually a lie. But:
    #
    # 1. std_string_view automatically coerces into std::filesystem::path.
    # 2. We know how to automatically python strings into std_string_view.
    #
    # So this gives us the best of both worlds.
    cdef _LogicalArray _from_file \
        "legate::experimental::io::kvikio::from_file" (
            std_string_view, const _Type&
        ) except+
    cdef void _to_file \
        "legate::experimental::io::kvikio::to_file" (
            std_string_view, const _LogicalArray&
        ) except+

    cdef _LogicalArray _from_file \
        "legate::experimental::io::kvikio::from_file" (
            std_string_view,
            const _Shape&,
            const _Type&,
            const std_vector[uint64_t]&,
            std_optional[std_vector[uint64_t]],
        ) except+
    cdef _LogicalArray _from_file \
        "legate::experimental::io::kvikio::from_file" (
            std_string_view,
            const _Shape&,
            const _Type&,
            const std_vector[uint64_t]&
        ) except+

    cdef void _to_file "legate::experimental::io::kvikio::to_file"(
        std_string_view,
        const _LogicalArray&,
        const std_vector[uint64_t]&,
        std_optional[std_vector[uint64_t]],
    ) except+
    cdef void _to_file "legate::experimental::io::kvikio::to_file"(
        std_string_view, const _LogicalArray&, const std_vector[uint64_t]&
    ) except+

    cdef _LogicalArray _from_file_by_offsets \
        "legate::experimental::io::kvikio::from_file_by_offsets" (
            std_string_view,
            const _Shape&,
            const _Type&,
            const std_vector[uint64_t]&,
            const std_vector[uint64_t]&,
        ) except+


cpdef LogicalArray from_file(object path, Type array_type)
cpdef void to_file(object path, LogicalArray array)

cpdef LogicalArray from_tiles(
    object path,
    object shape,
    Type array_type,
    tuple[uint64_t, ...] tile_shape,
    tuple[uint64_t, ...] tile_start = *
)
cpdef void to_tiles(
    object path,
    LogicalArray array,
    tuple[uint64_t, ...] tile_shape,
    tuple[uint64_t, ...] tile_start = *
)

cpdef LogicalArray from_tiles_by_offsets(
    object path,
    object shape,
    Type type,
    tuple[uint64_t, ...] offsets,
    tuple[uint64_t, ...] tile_shape,
)
