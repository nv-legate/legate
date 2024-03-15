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

import numpy as np
from numpy._typing import NDArray

from legate.core import Scalar
from legate.core.task import (
    InputArray,
    InputStore,
    OutputArray,
    OutputStore,
    task,
)


@task
def basic_task() -> None:
    pass


@task
def copy_store_task(in_store: InputStore, out_store: OutputStore) -> None:
    in_arr_np = np.asarray(in_store.get_inline_allocation())
    out_arr_np = np.asarray(out_store.get_inline_allocation())
    out_arr_np[:] = in_arr_np[:]


@task
def mixed_sum_task(
    arg1: InputArray, arg2: InputStore, out: OutputArray
) -> None:
    arr1_np = np.asarray(arg1.data().get_inline_allocation())
    arr2_np = np.asarray(arg2.get_inline_allocation())
    out_arr_np = np.asarray(out.data().get_inline_allocation())
    out_arr_np[:] = arr1_np + arr2_np


@task
def fill_task(out: OutputArray, val: Scalar) -> None:
    out_arr_np = np.asarray(out.data().get_inline_allocation())
    out_arr_np.fill(val)


@task
def copy_np_array_task(out: OutputStore, np_arr: NDArray) -> None:
    out_arr_np = np.asarray(out.get_inline_allocation())
    out_arr_np[:] = np_arr[:]
