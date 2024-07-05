# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
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
    """Create a NumPy array and a LogicalStore from it"""
    runtime = get_legate_runtime()
    dtype = legate_dtype.to_numpy_dtype()
    arr = func(shape, dtype)
    store = runtime.create_store_from_buffer(
        legate_dtype, arr.shape, arr, read_only=read_only
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
