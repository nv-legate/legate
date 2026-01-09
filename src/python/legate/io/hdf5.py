# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

__all__ = ("from_file", "from_file_batched", "to_file")

import math
import itertools
from typing import TYPE_CHECKING

import h5py  # type: ignore[import-untyped]

from ..core import LogicalArray, Type, get_legate_runtime

# This module is the "public" interface for this function, so import it purely
# to re-export it.
from ._lib.hdf5.hdf5_interface import from_file, to_file

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from ._lib.hdf5.hdf5_interface import Pathlike


def from_file_batched(
    path: Pathlike, dataset_name: str, chunk_size: Sequence[int]
) -> Generator[tuple[LogicalArray, tuple[int, ...]], None, None]:
    r"""Read a HDF5 file as a series of batches, yielding an array over each
    batch.

    If the array to read does not evenly divide into batches of `chunk_size`,
    then any chunks on the boundaries will be "clipped". For example, if the
    array on disk is size 5, but `chunk_size = (2,)`, then this routine will
    yield arrays of size `(2,)`, `(2,)` and `(1,)`, in that order.

    In other words, each array `arr` returned will always have dimensionality
    equal to `len(chunk_size)`, but `arr.volume <= chunk_size.volume`.

    The returned offsets may be used to "orient" the returned chunk in the
    context of the greater array. For example, given an array of size `(5, 5)`,
    with `chunk_size = (2, 2)`, the returned offsets would be:
    ```
    (0, 0)
    (0, 2)
    (0, 4)
    (2, 0)
    (2, 2)
    (2, 4)
    (4, 0)
    (4, 2)
    (4, 4)
    ```
    As illustrated in the example, the array is always traversed from low to
    high.

    Parameters
    ----------
    path : Pathlike
        The path to the HDF5 file. The file must exist on disk at the time of
        this call.
    dataset_name : str
        The name of the dataset to read from the HDF5 file.
    chunk_size : Sequence[int]
        The maximum dimensions of the chunks to read.

    Yields
    ------
    LogicalArray
        The array over the chunk of data.
    tuple[int, ...]
        A tuple containing the offsets of the returned array into the global
        shape of the on-disk array.

    Raises
    ------
    ValueError
        If `chunk_size` contains values <= 0.
    ValueError
        If the dimensionality of `chunk_size` does not match that of the
        dataset to be read.
    """
    if math.prod(chunk_size) <= 0:
        m = f"Invalid chunk size ({chunk_size}), must be >0"
        raise ValueError(m)

    runtime = get_legate_runtime()

    with h5py.File(path, mode="r") as f:
        dset = f[dataset_name]
        shape = dset.shape

        if len(shape) != len(chunk_size):
            m = (
                f"Dimensions of chunks ({len(chunk_size)}) must match "
                f"dimension of dataset ({len(shape)})."
            )
            raise ValueError(m)

        dtype = Type.from_numpy_dtype(dset.dtype)

        offsets = tuple(
            range(0, ext, chunk)
            for ext, chunk in zip(shape, chunk_size, strict=True)
        )

        for offs in itertools.product(*offsets):
            slices = tuple(
                slice(o, min(o + chunk, ext))
                for o, ext, chunk in zip(offs, shape, chunk_size, strict=True)
            )

            batch = dset[slices]
            store = runtime.create_store_from_buffer(
                dtype=dtype, shape=batch.shape, data=batch, read_only=True
            )
            yield LogicalArray.from_store(store), offs
