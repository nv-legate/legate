# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Zarr v2 IO helpers.

This module handles the narrow Zarr v2 metadata subset needed by Legate
directly so zarr-python is not a runtime dependency. zarr-python remains a
test-only dependency for compatibility checks. Zarr v3 metadata is detected but
not implemented here.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from ... import (
    LogicalArray,
    Type,
    VariantCode,
    get_legate_runtime,
    types as ty,
)
from ...data_interface import LogicalArrayLike, as_logical_array
from ...task import InputStore, task
from . import tile

_ZARR_V2_DISK_FORMAT = 2
_ZARR_V3_DISK_FORMAT = 3
_ZARR_V3_UNSUPPORTED_MSG = "Zarr format 3 support is not implemented yet"
_ZARR_CHUNK_BASE = 256 * 1024
_ZARR_CHUNK_MIN = 128 * 1024
_ZARR_CHUNK_MAX = 64 * 1024 * 1024
_ZARR_CHUNK_TARGET_TOLERANCE = 0.5
_ZARR_SUPPORTED_DTYPE_KINDS = frozenset("biufc")


class _ZarrV2Metadata(NamedTuple):
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    dtype: np.dtype[Any]
    has_codecs: bool


def _guess_zarr_v2_chunks(
    shape: tuple[int, ...], typesize: int
) -> tuple[int, ...]:
    """Return zarr-python-compatible default v2 chunks."""
    # Adapted from zarr-python's _guess_chunks heuristic. Kept locally so
    # write_array() does not require zarr-python at runtime.
    if typesize == 0:
        return shape

    ndims = len(shape)
    chunks = np.maximum(np.array(shape, dtype="=f8"), 1)
    dset_size = np.prod(chunks) * typesize
    target_size = _ZARR_CHUNK_BASE * (
        2 ** np.log10(dset_size / (1024.0 * 1024))
    )

    if target_size > _ZARR_CHUNK_MAX:
        target_size = _ZARR_CHUNK_MAX
    elif target_size < _ZARR_CHUNK_MIN:
        target_size = _ZARR_CHUNK_MIN

    idx = 0
    while True:
        chunk_bytes = np.prod(chunks) * typesize
        close_to_target = (
            abs(chunk_bytes - target_size) / target_size
            < _ZARR_CHUNK_TARGET_TOLERANCE
        )
        if (
            chunk_bytes < target_size or close_to_target
        ) and chunk_bytes < _ZARR_CHUNK_MAX:
            break

        if np.prod(chunks) == 1:
            break

        chunks[idx % ndims] = math.ceil(chunks[idx % ndims] / 2.0)
        idx += 1

    return tuple(int(chunk) for chunk in chunks)


def _normalize_zarr_v2_chunk(chunk: object) -> int:
    if isinstance(chunk, bool) or not isinstance(chunk, (int, np.integer)):
        msg = "chunks must be an int, a tuple of ints, or None"
        raise TypeError(msg)
    return int(chunk)


def _normalize_zarr_v2_chunks(
    chunks: tuple[object, ...] | None, shape: tuple[int, ...], typesize: int
) -> tuple[int, ...]:
    if chunks is None:
        return _guess_zarr_v2_chunks(shape, typesize)

    if len(chunks) > len(shape):
        msg = "too many dimensions in chunks"
        raise ValueError(msg)

    if len(chunks) < len(shape):
        chunks += shape[len(chunks) :]

    normalized: list[int] = []
    for shape_dim, chunk in zip(shape, chunks, strict=True):
        chunk_i = _normalize_zarr_v2_chunk(chunk)
        normalized.append(shape_dim if chunk_i == -1 else chunk_i)
    chunks = tuple(normalized)

    if any(chunk <= 0 for chunk in chunks):
        msg = "chunks must be positive integers or -1"
        raise ValueError(msg)

    return chunks


def _zarr_fill_value(dtype: np.dtype[Any]) -> object:
    fill_value = np.array(0, dtype=dtype).item()
    if isinstance(fill_value, complex):
        return [float(fill_value.real), float(fill_value.imag)]
    return fill_value


def _raise_if_unsupported_zarr_dtype(dtype: np.dtype[Any]) -> None:
    if dtype.kind not in _ZARR_SUPPORTED_DTYPE_KINDS:
        msg = f"Zarr IO only supports numeric and bool dtypes, got {dtype}"
        raise NotImplementedError(msg)


def _write_json_file(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=4, sort_keys=True)
        f.write("\n")


def _check_zarr_dir_can_be_overwritten(dirpath: Path) -> None:
    if not dirpath.exists():
        return
    if not dirpath.is_dir():
        msg = f"{dirpath} exists and is not a directory"
        raise NotADirectoryError(msg)
    if not any(dirpath.iterdir()):
        return
    if (dirpath / ".zarray").exists() or (dirpath / "zarr.json").exists():
        return

    msg = (
        f"{dirpath} already exists and is not an empty directory or a "
        "recognized Zarr array"
    )
    raise FileExistsError(msg)


def _write_zarr_v2_metadata(
    dirpath: Path,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    chunks: tuple[int, ...],
) -> None:
    _check_zarr_dir_can_be_overwritten(dirpath)
    if dirpath.exists():
        shutil.rmtree(dirpath)

    dirpath.mkdir(parents=True)
    _write_json_file(
        dirpath / ".zarray",
        {
            "chunks": list(chunks),
            "compressor": None,
            "dimension_separator": ".",
            "dtype": dtype.str,
            "fill_value": _zarr_fill_value(dtype),
            "filters": None,
            "order": "C",
            "shape": list(shape),
            "zarr_format": _ZARR_V2_DISK_FORMAT,
        },
    )
    _write_json_file(dirpath / ".zattrs", {})


def _read_json_file(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        msg = f"Expected {path} to contain a JSON object"
        raise TypeError(msg)
    return data


def _metadata_value(metadata: dict[str, Any], name: str) -> Any:
    try:
        return metadata[name]
    except KeyError as ex:
        msg = f"Zarr metadata is missing required key {name!r}"
        raise ValueError(msg) from ex


def _metadata_tuple(metadata: dict[str, Any], name: str) -> tuple[int, ...]:
    value = _metadata_value(metadata, name)
    if not isinstance(value, (list, tuple)):
        msg = f"Expected {name!r} in Zarr metadata to be a sequence"
        raise TypeError(msg)
    ret: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, np.integer)):
            msg = f"Expected {name!r} items in Zarr metadata to be integers"
            raise TypeError(msg)
        ret.append(int(item))
    return tuple(ret)


def _raise_if_unsupported_zarr_format(dirpath: Path) -> None:
    zarr_json_path = dirpath / "zarr.json"
    if not zarr_json_path.exists():
        return

    zarr_format = _read_json_file(zarr_json_path).get("zarr_format")
    if zarr_format is not None and int(zarr_format) >= _ZARR_V3_DISK_FORMAT:
        raise NotImplementedError(_ZARR_V3_UNSUPPORTED_MSG)


def _read_zarr_v2_metadata(dirpath: Path) -> _ZarrV2Metadata:
    zarray_path = dirpath / ".zarray"
    if not zarray_path.exists():
        _raise_if_unsupported_zarr_format(dirpath)
        msg = f"{dirpath} does not contain a Zarr format 2 array"
        raise FileNotFoundError(msg)

    metadata = _read_json_file(zarray_path)
    zarr_format = _metadata_value(metadata, "zarr_format")
    if isinstance(zarr_format, bool) or not isinstance(
        zarr_format, (int, np.integer)
    ):
        msg = "Expected 'zarr_format' in Zarr metadata to be an integer"
        raise TypeError(msg)
    if int(zarr_format) >= _ZARR_V3_DISK_FORMAT:
        raise NotImplementedError(_ZARR_V3_UNSUPPORTED_MSG)
    if int(zarr_format) != _ZARR_V2_DISK_FORMAT:
        msg = f"Only Zarr format {_ZARR_V2_DISK_FORMAT} arrays are supported"
        raise NotImplementedError(msg)

    if metadata.get("order", "C") != "C":
        msg = "Only Zarr arrays with C order are supported"
        raise NotImplementedError(msg)

    if metadata.get("dimension_separator", ".") != ".":
        msg = "Only Zarr arrays with '.' dimension_separator are supported"
        raise NotImplementedError(msg)

    shape = _metadata_tuple(metadata, "shape")
    chunks = _metadata_tuple(metadata, "chunks")
    if len(shape) != len(chunks):
        msg = "Zarr metadata 'shape' and 'chunks' must have the same rank"
        raise ValueError(msg)
    if any(dim < 0 for dim in shape):
        msg = "Zarr metadata 'shape' must contain non-negative integers"
        raise ValueError(msg)
    if any(chunk <= 0 for chunk in chunks):
        msg = "Zarr metadata 'chunks' must contain positive integers"
        raise ValueError(msg)

    return _ZarrV2Metadata(
        shape=shape,
        chunks=chunks,
        dtype=np.dtype(_metadata_value(metadata, "dtype")),
        has_codecs=(
            metadata.get("compressor") is not None
            or metadata.get("filters") not in (None, [])
        ),
    )


@task(variants=tuple(VariantCode))
def init_zarr_dir_task(
    _input_store: InputStore,
    dirpath: str,
    array_shape: tuple[int, ...],
    array_dtype: str,
    chunks: tuple[int, ...],
    use_default_chunks: bool,  # noqa: FBT001
) -> None:
    """Task to initialize Zarr directory structure (single-instance)."""
    chunks_arg = None if use_default_chunks else tuple(chunks)
    dtype = np.dtype(array_dtype)
    shape = tuple(int(dim) for dim in array_shape)
    normalized_chunks = _normalize_zarr_v2_chunks(
        chunks_arg, shape, dtype.itemsize
    )
    _write_zarr_v2_metadata(Path(dirpath), shape, dtype, normalized_chunks)


def _get_padded_shape(
    array_shape: tuple[int, ...], chunks: tuple[int, ...]
) -> tuple[tuple[int, ...], bool]:
    r"""Get a padded array that has a shape divisible by the chunks.

    Parameters
    ----------
    array_shape : tuple[int, ...]
        The array shape.
    chunks : tuple[int, ...]
        The Zarr chunks.

    Return
    ------
    tuple[int, ...]
        The possibly padded array shape.
    bool
        True if the array shape was padded, False otherwise.
    """
    if all(s % c == 0 for s, c in zip(array_shape, chunks, strict=True)):
        return array_shape, False

    return (
        tuple(
            math.ceil(s / c) * c
            for s, c in zip(array_shape, chunks, strict=True)
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

    if not ary.shape:
        msg = "Cannot write a 0-dimensional (scalar) array to Zarr"
        raise ValueError(msg)

    dirpath = Path(dirpath)
    runtime = get_legate_runtime()

    array_shape = tuple(ary.shape)
    array_dtype_np = ary.type.to_numpy_dtype()
    _raise_if_unsupported_zarr_dtype(array_dtype_np)
    array_dtype = str(array_dtype_np)

    if isinstance(chunks, bool):
        msg = "chunks must be an int, a tuple of ints, or None"
        raise TypeError(msg)

    use_default_chunks = chunks is None
    if chunks is None:
        chunks_arg: tuple[int, ...] = (-1,)
    elif isinstance(chunks, (int, np.integer)):
        chunks_arg = (int(chunks),) * len(array_shape)
    else:
        try:
            chunks_arg = tuple(
                _normalize_zarr_v2_chunk(chunk) for chunk in chunks
            )
        except TypeError as ex:
            msg = "chunks must be an int, a tuple of ints, or None"
            raise TypeError(msg) from ex

    if any(chunk < -1 or chunk == 0 for chunk in chunks_arg):
        msg = "chunks must be positive integers or -1"
        raise ValueError(msg)

    _check_zarr_dir_can_be_overwritten(dirpath)

    # Create and execute a manual task with a single instance
    # This ensures the zarr directory is created only once across all ranks
    manual_task = runtime.create_manual_task(
        init_zarr_dir_task.library,
        init_zarr_dir_task.task_id,
        (1,),  # Single instance
    )
    manual_task.add_input(ary.data)
    manual_task.add_scalar_arg(str(dirpath), ty.string_type)
    manual_task.add_scalar_arg(array_shape, (ty.int64,))
    manual_task.add_scalar_arg(array_dtype, ty.string_type)
    manual_task.add_scalar_arg(chunks_arg, (ty.int64,))
    manual_task.add_scalar_arg(use_default_chunks, ty.bool_)
    manual_task.execute()
    runtime.issue_execution_fence(block=True)

    metadata = _read_zarr_v2_metadata(dirpath)

    # TODO: minimize the copy needed when padding
    shape, padded = _get_padded_shape(metadata.shape, metadata.chunks)
    if padded:
        padded_ary = runtime.create_array(
            Type.from_numpy_dtype(metadata.dtype), shape
        )
        padded_ary.fill(0)
        sliced = padded_ary[tuple(map(slice, metadata.shape))]
        runtime.issue_copy(sliced.data, ary.data)
        ary = padded_ary
    tile.to_tiles(path=dirpath, array=ary, tile_shape=metadata.chunks)


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
    metadata = _read_zarr_v2_metadata(dirpath)
    _raise_if_unsupported_zarr_dtype(metadata.dtype)

    if metadata.has_codecs:
        msg = "compressor and filters aren't supported"
        raise NotImplementedError(msg)

    shape, padded = _get_padded_shape(metadata.shape, metadata.chunks)
    ret = tile.from_tiles(
        path=dirpath,
        shape=shape,
        array_type=Type.from_numpy_dtype(metadata.dtype),
        tile_shape=metadata.chunks,
    )
    if padded:
        ret = ret[tuple(slice(s) for s in metadata.shape)]
    return ret
