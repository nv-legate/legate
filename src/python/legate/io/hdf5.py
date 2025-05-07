# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
import itertools
from typing import TYPE_CHECKING

import h5py  # type: ignore # noqa: PGH003
import zarr  # type: ignore # noqa: PGH003
import fsspec  # type: ignore # noqa: PGH003
from kerchunk.hdf import SingleHdf5ToZarr  # type: ignore # noqa: PGH003

from ..core import Type
from ..core.experimental.io.tile import from_tiles_by_offsets
from ..core.experimental.io.zarr import _get_padded_shape

if TYPE_CHECKING:
    from pathlib import Path

    from ..core import LogicalArray

__all__ = ("from_file", "kerchunk_read")

# This module is the "public" interface for this function, so import it purely
# to re-export it.
from ._lib.hdf5.hdf5_interface import from_file


def _get_virtual_dataset_names(filepath: str) -> set[str]:
    ret = set()

    def visitor(name: str, obj: h5py.Dataset | h5py.Group) -> None:  # type: ignore[no-any-unimported]
        if getattr(obj, "is_virtual", False):
            ret.add(name)

    with h5py.File(filepath, mode="r") as f:
        f.visititems(visitor)

    return ret


def kerchunk_read(filepath: Path | str, dataset_name: str) -> LogicalArray:
    r"""Read an HDF5 array from disk using Kerchunk and KvikIO.

    We use Kerchunk's ``SingleHdf5ToZarr`` to find the data chunks embedded
    in the hdf5 file. If it fails for any reason, this function fails as well.

    Notes
    -----
    The returned array might be a view of an underlying array that has been
    padded in order to make its shape divisible by the shape of the hdf5
    chunks on disk.

    Parameters
    ----------
    filepath
        File path to the hdf5 file.
    dataset_name
        Name/path of the dataset. This must reference a single array, thus
        make sure to use the full path to the array inside the HDF5 file.

    Return
    ------
        The Legate array read from disk.
    """
    filepath = str(filepath)

    # TODO: look for already generated kerchunk annotations
    annotations = SingleHdf5ToZarr(filepath, inline_threshold=0).translate()

    # Load annotations
    zarr_group = zarr.open(
        fsspec.get_mapper(
            "reference://",
            fo=annotations,  # codespell:ignore fo
        )
    )
    if not isinstance(zarr_group, zarr.Group):
        msg = (
            "root of the HDF5 file must be a dataset, "
            f"found {type(zarr_group)}"
        )
        raise TypeError(msg)

    # Make sure `dataset_name` refer to an array and not a dataset
    zarr_ary = zarr_group[dataset_name]
    if isinstance(zarr_ary, zarr.Group):
        # In this case, the user is giving us a string name for the dataset, so
        # the "kind" of error here is that the name of the dataset is not
        # correct. We detect this by way of a type error (the returned object
        # is not the same type that we expect), but the actual error is that
        # the value of the thing the user passed to us is wrong.
        #
        # This is why we silence ruff below.
        msg = (
            "dataset_name must reference an array, please use the full "
            f"path to an array in the dataset: {list(zarr_ary.arrays())}"
        )
        raise ValueError(msg)  # noqa: TRY004

    if zarr_ary.compressor is not None:
        msg = "compressor isn't supported"
        raise NotImplementedError(msg)

    # Extract offset and bytes for each chunk
    refs = annotations["refs"]
    offsets = []
    tile_nbytes = math.prod(zarr_ary.chunks) * zarr_ary.itemsize
    dims = (
        range(math.ceil(s / c))
        for s, c in zip(zarr_ary.shape, zarr_ary.chunks, strict=True)
    )
    for chunk_coord in itertools.product(*dims):
        key = zarr_ary._chunk_key(chunk_coord)  # noqa: SLF001
        try:
            _, offset, nbytes = refs[key]  # pyright: ignore[reportArgumentType,reportCallIssue]
        except KeyError:
            if dataset_name in _get_virtual_dataset_names(filepath):
                msg = f"Virtual dataset isn't supported: {dataset_name}"
                raise NotImplementedError(msg)
            raise
        offsets.append(offset)
        assert tile_nbytes == nbytes

    shape, padded = _get_padded_shape(zarr_ary)
    out = from_tiles_by_offsets(
        path=filepath,
        shape=shape,
        type=Type.from_numpy_dtype(zarr_ary.dtype),
        offsets=tuple(offsets),
        tile_shape=zarr_ary.chunks,
    )
    if padded:
        out = out[tuple(slice(s) for s in zarr_ary.shape)]
    return out
