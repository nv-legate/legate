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


from enum import IntEnum
from typing import Any, Tuple

import legate.core.types as ty
from legate.core import Rect, ReductionOp, Store

from .lib import user_context as context, user_lib  # type: ignore


class OpCode(IntEnum):
    COLLECTIVE = user_lib.cffi.COLLECTIVE


def create_int64_store(shape: Tuple[Any, ...]) -> Store:
    store = context.create_store(ty.int64, shape=shape)
    # call empty function on the store to make Legion to think
    # that we initialized the store
    task = context.create_auto_task(
        OpCode.COLLECTIVE,
    )
    task.add_output(store)
    task.execute()
    return store


def collective_test(
    store: Store, shape: Tuple[Any, ...], tile_shape: Tuple[Any, ...]
) -> None:
    assert store.ndim == len(shape)
    if store.shape != shape:
        diff = len(shape) - store.ndim
        for dim in range(diff):
            store = store.promote(dim, shape[dim])
        for dim in range(len(shape)):
            if store.shape[dim] != shape[dim]:
                if store.shape[dim] != 1:
                    raise ValueError(
                        f"Shape did not match along dimension {dim} "
                        "and the value is not equal to 1"
                    )
                store = store.project(dim, 0).promote(dim, shape[dim])

    store_partition = store.partition_by_tiling(tile_shape)
    launch_shape = store_partition.partition.color_shape

    task = context.create_manual_task(
        OpCode.COLLECTIVE,
        launch_domain=Rect(launch_shape),
    )
    task.add_input(store_partition)
    # we use errors from the mapper to check correct behavior
    task.execute()


def collective_test_matvec(rhs1: Store, rhs2: Store, lhs: Store) -> None:
    shape = rhs1.shape
    rhs2 = rhs2.promote(0, shape[0])
    lhs = lhs.promote(1, shape[1])
    task = context.create_auto_task(
        OpCode.COLLECTIVE,
    )
    task.add_reduction(lhs, ReductionOp.ADD)
    task.add_input(rhs1)
    task.add_input(rhs2)
    task.add_alignment(lhs, rhs1)
    task.add_alignment(lhs, rhs2)
    task.execute()
