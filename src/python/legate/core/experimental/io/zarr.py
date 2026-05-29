# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from pathlib import Path

import zarr  # type: ignore # noqa: PGH003
import numpy as np
import zarr.core  # type: ignore # noqa: PGH003

from ... import (
    LogicalStore,
    Type,
    VariantCode,
    get_legate_runtime,
    types as ty,
)
from ...data_interface import LogicalStoreLike, as_logical_store
from ...task import InputStore, task
from . import tile


@task(variants=tuple(VariantCode))
def init_zarr_dir_task(
    _input_store: InputStore,
    dirpath: str,
    array_shape: tuple[int, ...],
    array_dtype: str,
    chunks: tuple[int, ...],
) -> None:
    """Task to initialize Zarr directory structure (single-instance)."""
    # Convert chunks sentinel (-1,) back to None
    chunks_arg = tuple(chunks) if chunks[0] > 0 else None
    zarr.open_array(
        dirpath,
        shape=array_shape,
        dtype=np.dtype(array_dtype),
        mode="w",
        chunks=chunks_arg,
        compressor=None,
    )


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
    st: LogicalStoreLike,
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
    st : LogicalStoreLike
       The Legate array-like object to write.
    dirpath : Path | str
        Root directory of the tile files.
    chunks : int | tuple[int, ...] | None
        The shape of each tile.
    """
    store = as_logical_store(st)

    if int(zarr.__version__.split(".")[0]) >= 3:  # noqa: PLR2004
        msg = "Zarr v3 support is not implemented yet"
        raise NotImplementedError(msg)

    if not store.shape:
        msg = "Cannot write a 0-dimensional (scalar) array to Zarr"
        raise ValueError(msg)

    dirpath = Path(dirpath)
    runtime = get_legate_runtime()

    store_shape = store.shape
    store_dtype = str(store.type.to_numpy_dtype())

    if chunks is None:
        chunks_arg: tuple[int, ...] = (-1,)
    elif isinstance(chunks, int):
        chunks_arg = (chunks,) * len(store_shape)
    else:
        chunks_arg = tuple(chunks)

    # Create and execute a manual task with a single instance
    # This ensures the zarr directory is created only once across all ranks
    manual_task = runtime.create_manual_task(
        init_zarr_dir_task.library,
        init_zarr_dir_task.task_id,
        (1,),  # Single instance
    )
    manual_task.add_input(store)
    manual_task.add_scalar_arg(str(dirpath), ty.string_type)
    manual_task.add_scalar_arg(store_shape, (ty.int64,))
    manual_task.add_scalar_arg(store_dtype, ty.string_type)
    manual_task.add_scalar_arg(chunks_arg, (ty.int64,))
    manual_task.execute()
    runtime.issue_execution_fence(block=True)

    # Now read the metadata that was created
    zarr_ary = zarr.open_array(dirpath, mode="r+")

    # TODO: minimize the copy needed when padding
    shape, padded = _get_padded_shape(zarr_ary)
    if padded:
        padded_st = runtime.create_store(
            Type.from_numpy_dtype(zarr_ary.dtype), shape
        )
        padded_st.fill(0)
        sliced = padded_st[tuple(map(slice, zarr_ary.shape))]
        runtime.issue_copy(sliced, store)
        store = padded_st
    tile.to_tiles(path=dirpath, store=store, tile_shape=zarr_ary.chunks)


def read_array(dirpath: Path | str) -> LogicalStore:
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
    LogicalStore
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
        store_type=Type.from_numpy_dtype(zarr_ary.dtype),
        tile_shape=zarr_ary.chunks,
    )
    if padded:
        ret = ret[tuple(slice(s) for s in zarr_ary.shape)]
    return ret
