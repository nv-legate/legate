# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

from typing import Any

try:
    import cupy
except ModuleNotFoundError:
    cupy = None

import numpy as np
from numpy._typing import NDArray

from legate.core import InlineAllocation, Scalar
from legate.core._ext.task.util import KNOWN_VARIANTS
from legate.core.task import (
    InputArray,
    InputStore,
    OutputArray,
    OutputStore,
    ReductionStore,
    task,
)


def asarray(alloc: InlineAllocation) -> NDArray[Any]:
    try:
        arr = np.asarray(alloc)
    except ValueError as exc:
        if cupy is None:
            raise RuntimeError("Need to install cupy for GPU variant") from exc
        arr = cupy.asarray(alloc)
    return arr


@task(variants=tuple(KNOWN_VARIANTS))
def basic_task() -> None:
    pass


@task(variants=tuple(KNOWN_VARIANTS))
def zeros_task(out: OutputStore) -> None:
    out_arr = asarray(out.get_inline_allocation())
    out_arr[:] = np.zeros(out_arr.shape)


@task(variants=tuple(KNOWN_VARIANTS))
def copy_store_task(in_store: InputStore, out_store: OutputStore) -> None:
    in_arr_np = asarray(in_store.get_inline_allocation())
    out_arr_np = asarray(out_store.get_inline_allocation())
    out_arr_np[:] = in_arr_np[:]


@task(variants=tuple(KNOWN_VARIANTS))
def partition_to_store_task(partition: InputStore, out: OutputStore) -> None:
    arr = asarray(partition.get_inline_allocation())
    out_arr = asarray(out.get_inline_allocation())
    out_arr[:] = out_arr + arr


@task(variants=tuple(KNOWN_VARIANTS))
def mixed_sum_task(
    arg1: InputArray, arg2: InputStore, out: OutputArray
) -> None:
    arr1_np = asarray(arg1.data().get_inline_allocation())
    arr2_np = asarray(arg2.get_inline_allocation())
    out_arr_np = asarray(out.data().get_inline_allocation())
    out_arr_np[:] = arr1_np + arr2_np


@task(variants=tuple(KNOWN_VARIANTS))
def fill_task(out: OutputArray, val: Scalar) -> None:
    out_arr_np = asarray(out.data().get_inline_allocation())
    out_arr_np.fill(val)


@task(variants=tuple(KNOWN_VARIANTS))
def copy_np_array_task(out: OutputStore, np_arr: NDArray[Any]) -> None:
    out_arr_np = asarray(out.get_inline_allocation())
    try:
        out_arr_np[:] = np_arr[:]
    except ValueError:
        out_arr_np[:] = cupy.asarray(np_arr)[:]


@task(variants=tuple(KNOWN_VARIANTS))
def array_sum_task(store: InputStore, out: ReductionStore) -> None:
    store_arr = asarray(store.get_inline_allocation())
    out_arr = asarray(out.get_inline_allocation())
    out_arr[:] = out_arr + store_arr.sum()
