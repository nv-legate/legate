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

from typing import Any, Callable

import numpy as np

from legate.core import LogicalStore, get_legate_runtime, types as ty


def create_np_array_and_store(
    legate_dtype: ty.Type,
    shape: tuple[int, ...],
    read_only: bool = False,
    func: Callable[
        [tuple[int, ...], np.dtype[Any]], np.ndarray[Any, Any]
    ] = np.ndarray,
) -> tuple[np.ndarray[Any, Any], LogicalStore]:
    """Create a NumPy array and a LogicalStore from it."""
    runtime = get_legate_runtime()
    dtype = legate_dtype.to_numpy_dtype()
    arr = func(shape, dtype)
    # Legate converts its fixed array types into array types in NumPy, which
    # then flattens the types and instead add another dimension to the arrays
    store_dtype = (
        legate_dtype.element_type
        if isinstance(legate_dtype, ty.FixedArrayType)
        else legate_dtype
    )
    store = runtime.create_store_from_buffer(
        store_dtype, arr.shape, arr, read_only=read_only
    )
    return arr, store


def random_array_and_store(
    shape: tuple[int, ...], read_only: bool = False
) -> tuple[np.ndarray[Any, Any], LogicalStore]:
    runtime = get_legate_runtime()
    arr = np.random.rand(*shape)
    store = runtime.create_store_from_buffer(
        ty.float64, arr.shape, arr, read_only=read_only
    )
    return arr, store


def zero_array_and_store(
    legate_dtype: ty.Type, shape: tuple[int, ...], read_only: bool = False
) -> tuple[np.ndarray[Any, Any], LogicalStore]:
    return create_np_array_and_store(
        legate_dtype, shape, read_only, func=np.zeros
    )


def empty_array_and_store(
    legate_dtype: ty.Type, shape: tuple[int, ...], read_only: bool = False
) -> tuple[np.ndarray[Any, Any], LogicalStore]:
    return create_np_array_and_store(
        legate_dtype, shape, read_only, func=np.empty
    )


def create_initialized_store(
    dtype: ty.Type, shape: tuple[int, ...], val: Any
) -> LogicalStore:
    runtime = get_legate_runtime()
    store = runtime.create_store(dtype, shape)
    runtime.issue_fill(store, val)
    return store


def create_random_points(
    shape: tuple[int, ...], dimensions: tuple[int, ...], no_duplicates: bool
) -> tuple[tuple[np.ndarray[Any, np.dtype[Any]], ...], LogicalStore]:
    store_vol = np.prod(shape)
    tgt_vol = np.prod(dimensions)
    ndim = len(dimensions)
    # When the volume of the store to create is bigger than the volume of the
    # domain, the store must contain the same point more than once, which leads
    # to non-deterministic behavior for scatter copies; i.e., the scatter copy
    # would essentially do `arr[ind] = val` for more than one `val` in
    # parallel and thus the semantics is ill-defined.
    if no_duplicates and store_vol > tgt_vol:
        msg = "The volume of the store must be smaller than the target volume"
        raise ValueError(msg)

    idx = list(np.indices(dimensions))
    points = np.stack(idx, axis=-1).reshape(tgt_vol, ndim)
    to_select = np.random.permutation(store_vol) % tgt_vol
    shuffled = points[to_select, :]

    store = get_legate_runtime().create_store_from_buffer(
        ty.point_type(ndim), shape, shuffled, read_only=True
    )

    coords = tuple(
        coord.squeeze()
        for coord in np.split(shuffled.reshape((*shape, ndim)), ndim, axis=-1)
    )
    return coords, store
