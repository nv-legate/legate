# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from libc.stdint cimport uint64_t
from libcpp.utility cimport move as std_move
from libcpp.vector cimport vector as std_vector

from .....data_interface import as_logical_store

from ....._ext.cython_libcpp.string_view cimport (
    std_string_view,
    std_string_view_from_py,
)
from ....data.logical_store cimport LogicalStore, _LogicalStore
from ....type.types cimport Type
from ....utilities.utils cimport std_vector_from_iterable


cpdef LogicalStore from_file(object path, Type store_type):
    cdef std_string_view cpp_path
    cdef _LogicalStore ret

    cpp_path = std_string_view_from_py(str(path))

    with nogil:
        ret = _from_file(cpp_path, store_type._handle)

    return LogicalStore.from_handle(std_move(ret))


cpdef void to_file(object path, object store):
    st = as_logical_store(store)
    _logical_store_to_file(path, st)


cdef void _logical_store_to_file(object path, LogicalStore store):
    cdef std_string_view cpp_path = std_string_view_from_py(str(path))

    with nogil:
        _to_file(cpp_path, store._handle)


cpdef LogicalStore from_tiles(
    object path,  # Pathlike
    object shape,  # Shapelike
    Type store_type,
    tile_shape: tuple[uint64_t, ...],
    tile_start: tuple[uint64_t, ...] = None
):
    r"""Read multiple tiles from disk into a store using KvikIO.

    The store shape must be divisible by the tile shape.

    Parameters
    ----------
    path : Pathlike
        Root directory of the tile files.
    shape : Shapelike
        The shape of the store.
    store_type : Type
        The datatype of the store.
    tile_shape : tuple[int, ...]
        The shape of each tile.
    tile_start : tuple[int, ...] | None
        The start coordinate of the tiles.

    Returns
    -------
    LogicalStore
        The store read from disk.
    """
    cdef std_string_view cpp_path
    cdef _Shape cpp_shape
    cdef std_vector[uint64_t] cpp_tile_shape
    cdef std_optional[std_vector[uint64_t]] cpp_tile_start

    cpp_path = std_string_view_from_py(str(path))
    cpp_shape = Shape.from_shape_like(shape)
    cpp_tile_shape = std_vector_from_iterable[uint64_t](tile_shape)

    if tile_start is None:
        cpp_tile_start = std_optional[std_vector[uint64_t]]()
    else:
        cpp_tile_start = std_vector_from_iterable[uint64_t](tile_start)

    cdef _LogicalStore ret

    with nogil:
        ret = _from_file(
            cpp_path,
            std_move(cpp_shape),
            store_type._handle,
            std_move(cpp_tile_shape),
            std_move(cpp_tile_start)
        )

    return LogicalStore.from_handle(std_move(ret))


cpdef void to_tiles(
    object path,
    LogicalStore store,
    tile_shape: tuple[uint64_t, ...],
    tile_start: tuple[uint64_t, ...] = None,
):
    r"""Write a store as multiple tiles to disk using KvikIO.

    The store shape must be divisible by the tile shape.

    Parameters
    ----------
    path : Pathlike
        The path to write to.
    store : LogicalStore
        The store to write.
    tile_shape : tuple[int, ...]
        The shape of each tile.
    tile_start : tuple[int, ...] | None
        The start coordinate of the tiles.
    """
    cdef std_string_view cpp_path
    cdef std_vector[uint64_t] cpp_tile_shape
    cdef std_optional[std_vector[uint64_t]] cpp_tile_start

    cpp_path = std_string_view_from_py(str(path))
    cpp_tile_shape = std_vector_from_iterable[uint64_t](tile_shape)
    if tile_start is None:
        cpp_tile_start = std_optional[std_vector[uint64_t]]()
    else:
        cpp_tile_start = std_vector_from_iterable[uint64_t](tile_start)

    with nogil:
        _to_file(
            cpp_path,
            store._handle,
            std_move(cpp_tile_shape),
            std_move(cpp_tile_start)
        )

cpdef LogicalStore from_tiles_by_offsets(
    object path,
    object shape,
    Type type,
    offsets: tuple[uint64_t, ...],
    tile_shape: tuple[uint64_t, ...]
):
    r"""Read multiple tiles from a single file into a store using KvikIO

    The store shape must be divisible by the tile shape.

    Parameters
    ----------
    path : Pathlike
        The path to the file to read.
    shape : Shapelike
        The shape of the store.
    type : Type
        The type of the store.
    offsets : tuple[int, ...]
        The offset of each tile in the file (in bytes). If the store is
        multi-dimensional, the offsets for its tiles must be listed in C-order.
    tile_shape : tuple[int, ...]
        The shape of the tiles (all tiles have the same shape).
    """
    cdef std_string_view cpp_path
    cdef std_vector[uint64_t] cpp_offsets
    cdef std_vector[uint64_t] cpp_tile_shape
    cdef _Shape cpp_shape

    cpp_path = std_string_view_from_py(str(path))
    cpp_offsets = std_vector_from_iterable[uint64_t](offsets)
    cpp_tile_shape = std_vector_from_iterable[uint64_t](tile_shape)
    cpp_shape = Shape.from_shape_like(shape)

    cdef _LogicalStore ret

    with nogil:
        ret = _from_file_by_offsets(
            cpp_path,
            std_move(cpp_shape),
            type._handle,
            std_move(cpp_offsets),
            std_move(cpp_tile_shape)
        )

    return LogicalStore.from_handle(std_move(ret))
