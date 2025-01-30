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
from __future__ import annotations

from libc.stdint cimport uint64_t
from libcpp.utility cimport move as std_move
from libcpp.vector cimport vector as std_vector

from ....._ext.cython_libcpp.string_view cimport (
    string_view as std_string_view,
    string_view_from_py as std_string_view_from_py,
)
from ....data.logical_array cimport LogicalArray, _LogicalArray
from ....type.types cimport Type
from ....utilities.utils cimport std_vector_from_iterable


cpdef LogicalArray from_file(object path, Type array_type):
    cdef std_string_view cpp_path
    cdef _LogicalArray ret

    cpp_path = std_string_view_from_py(str(path))

    with nogil:
        ret = _from_file(cpp_path, array_type._handle)

    return LogicalArray.from_handle(std_move(ret))

cpdef void to_file(object path, LogicalArray array):
    cdef std_string_view cpp_path = std_string_view_from_py(str(path))

    with nogil:
        _to_file(cpp_path, array._handle)


cpdef LogicalArray from_tiles(
    object path,  # Pathlike
    object shape,  # Shapelike
    Type array_type,
    tile_shape: tuple[uint64_t, ...],
    tile_start: tuple[uint64_t, ...] = None
):
    r"""Read multiple tiles from disk into an array using KvikIO.

    The array shape must be divisible by the tile shape.

    Parameters
    ----------
    path : Pathlike
        Root directory of the tile files.
    shape : Shapelike
        The shape of the array.
    array_type : Type
        The datatype of the array.
    tile_shape : tuple[int, ...]
        The shape of each tile.
    tile_start : tuple[int, ...] | None
        The start coordinate of the tiles.

    Returns
    -------
    LogicalArray
        The array read from disk.
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

    cdef _LogicalArray ret

    with nogil:
        ret = _from_file(
            cpp_path,
            std_move(cpp_shape),
            array_type._handle,
            std_move(cpp_tile_shape),
            std_move(cpp_tile_start)
        )

    return LogicalArray.from_handle(std_move(ret))


cpdef void to_tiles(
    object path,
    LogicalArray array,
    tile_shape: tuple[uint64_t, ...],
    tile_start: tuple[uint64_t, ...] = None,
):
    r"""Write an array as multiple tiles to disk using KvikIO.

    The array shape must be divisible by the tile shape.

    Parameters
    ----------
    path : Pathlike
        The path to write to.
    array : LogicalArray
        The array to write.
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
            array._handle,
            std_move(cpp_tile_shape),
            std_move(cpp_tile_start)
        )

cpdef LogicalArray from_tiles_by_offsets(
    object path,
    object shape,
    Type type,
    offsets: tuple[uint64_t, ...],
    tile_shape: tuple[uint64_t, ...]
):
    r"""Read multiple tiles from a single file into an array using KvikIO

    The array shape must be divisible by the tile shape.

    Parameters
    ----------
    path : Pathlike
        The path to the file to read.
    shape : Shapelike
        The shape of the array.
    type : Type
        The type of the array.
    offsets : tuple[int, ...]
        The offset of each tile in the file (in bytes). If the array is
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

    cdef _LogicalArray ret

    with nogil:
        ret = _from_file_by_offsets(
            cpp_path,
            std_move(cpp_shape),
            type._handle,
            std_move(cpp_offsets),
            std_move(cpp_tile_shape)
        )

    return LogicalArray.from_handle(std_move(ret))
