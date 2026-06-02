# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint64_t
from libcpp.optional cimport optional as std_optional
from libcpp.vector cimport vector as std_vector

from ....._ext.cython_libcpp.string_view cimport std_string_view
from ....data.logical_store cimport LogicalStore, _LogicalStore
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
    cdef void _to_file \
        "legate::experimental::io::kvikio::to_file" (
            std_string_view, const _LogicalStore&
        ) except+

    cdef void _to_file "legate::experimental::io::kvikio::to_file"(
        std_string_view,
        const _LogicalStore&,
        const std_vector[uint64_t]&,
        std_optional[std_vector[uint64_t]],
    ) except+
    cdef void _to_file "legate::experimental::io::kvikio::to_file"(
        std_string_view, const _LogicalStore&, const std_vector[uint64_t]&
    ) except+

    cdef _LogicalStore _from_file \
        "legate::experimental::io::kvikio::from_file" (
            std_string_view, const _Type&
        ) except+

    cdef _LogicalStore _from_file \
        "legate::experimental::io::kvikio::from_file" (
            std_string_view,
            const _Shape&,
            const _Type&,
            const std_vector[uint64_t]&,
            std_optional[std_vector[uint64_t]],
        ) except+
    cdef _LogicalStore _from_file \
        "legate::experimental::io::kvikio::from_file" (
            std_string_view,
            const _Shape&,
            const _Type&,
            const std_vector[uint64_t]&
        ) except+

    cdef _LogicalStore _from_file_by_offsets \
        "legate::experimental::io::kvikio::from_file_by_offsets" (
            std_string_view,
            const _Shape&,
            const _Type&,
            const std_vector[uint64_t]&,
            const std_vector[uint64_t]&,
        ) except+


cpdef LogicalStore from_file(object path, Type store_type)
cdef void _logical_store_to_file(object path, LogicalStore store)

cpdef LogicalStore from_tiles(
    object path,
    object shape,
    Type store_type,
    tuple[uint64_t, ...] tile_shape,
    tuple[uint64_t, ...] tile_start = *
)
cpdef void to_tiles(
    object path,
    LogicalStore store,
    tuple[uint64_t, ...] tile_shape,
    tuple[uint64_t, ...] tile_start = *
)

cpdef LogicalStore from_tiles_by_offsets(
    object path,
    object shape,
    Type type,
    tuple[uint64_t, ...] offsets,
    tuple[uint64_t, ...] tile_shape,
)
