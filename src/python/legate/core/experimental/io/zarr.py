# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from pathlib import Path

import zarr  # type: ignore # noqa: PGH003
import zarr.core  # type: ignore # noqa: PGH003

from ... import LogicalArray, Type, get_legate_runtime
from ...data_interface import LogicalArrayLike, as_logical_array
from . import tile


def _get_padded_shape(  # type: ignore # noqa: PGH003
    zarr_ary: zarr.Array,
) -> tuple[tuple[int, ...], bool]:
    r"""Get a padded array that has a shape divisible by ``zarr_ary.chunks``.

    Parameters
    ----------
    zarr_ary : zarr.Array
       The Zarr array

    Return
    ------
    tuple[int, ...]
        The possibly padded array shape.
    bool
        True if the array shape was padded, False otherwise.
    """
    if all(
        s % c == 0
        for s, c in zip(zarr_ary.shape, zarr_ary.chunks, strict=True)
    ):
        return zarr_ary.shape, False

    return (
        tuple(
            math.ceil(s / c) * c
            for s, c in zip(zarr_ary.shape, zarr_ary.chunks, strict=True)
        ),
        True,
    )


def write_array(
    ary: LogicalArrayLike,
    dirpath: Path | str,
    chunks: int | tuple[int, ...] | None = None,
) -> None:
    """Write a Legate array to disk using the Zarr format.

    Notes
    -----
    The array is padded to make its shape divisible by chunks (if not already).
    This involves copying the whole array, which can be expensive both in
    terms of performance and memory usage.

    Parameters
    ----------
    ary : LogicalArrayLike
       The Legate array-like object to write.
    dirpath : Path | str
        Root directory of the tile files.
    chunks : int | tuple[int, ...] | None
        The shape of each tile.
    """
    ary = as_logical_array(ary)

    dirpath = Path(dirpath)

    # We use Zarr to write the meta data
    zarr_ary = zarr.open_array(
        dirpath,
        shape=ary.shape,
        dtype=ary.type.to_numpy_dtype(),
        mode="w",
        # pyright hallucinates types for these zarr parameters
        chunks=chunks,  # pyright: ignore[reportArgumentType]
        compressor=None,  # pyright: ignore[reportArgumentType]
    )
    # TODO: minimize the copy needed when padding
    shape, padded = _get_padded_shape(zarr_ary)
    if padded:
        runtime = get_legate_runtime()
        padded_ary = runtime.create_array(
            Type.from_numpy_dtype(zarr_ary.dtype), shape
        )
        padded_ary.fill(0)
        sliced = padded_ary[tuple(map(slice, zarr_ary.shape))]
        runtime.issue_copy(sliced.data, ary.data)
        ary = padded_ary
    tile.to_tiles(path=dirpath, array=ary, tile_shape=zarr_ary.chunks)


def read_array(dirpath: Path | str) -> LogicalArray:
    """Read a Zarr array from disk in to a Legate array.

    Notes
    -----
    The returned array might be a view of an underlying array that has been
    padded in order to make its shape divisible by the shape of the Zarr
    chunks on disk.

    Parameters
    ----------
    dirpath : Path | str
        Root directory of the tile files.

    Return
    ------
    LogicalArray
        The Legate array read from disk.
    """
    dirpath = Path(dirpath)

    # We use Zarr to read the meta data
    zarr_ary = zarr.open_array(dirpath, mode="r")

    # Zarr v3 changed the way the data is stored on disk. Instead of a single
    # directory with a bunch of files, it now stores the files hierarchically
    # according to the spec here
    # https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html. I haven't
    # had the time to fully look into it, so just bail here.
    if hasattr(zarr_ary, "metadata") and zarr_ary.metadata.zarr_format >= 3:  # noqa: PLR2004
        m = "Zarr v3 support is not implemented yet"
        raise NotImplementedError(m)

    if zarr_ary.compressor is not None:
        msg = "compressor isn't supported"
        raise NotImplementedError(msg)

    shape, padded = _get_padded_shape(zarr_ary)
    ret = tile.from_tiles(
        path=dirpath,
        shape=shape,
        array_type=Type.from_numpy_dtype(zarr_ary.dtype),
        tile_shape=zarr_ary.chunks,
    )
    if padded:
        ret = ret[tuple(slice(s) for s in zarr_ary.shape)]
    return ret
