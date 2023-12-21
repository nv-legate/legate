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


import pytest
from tree_reduce import user_context, user_lib

import legate.core.types as ty
from legate.core import Rect


def test_tree_reduce_normal():
    num_tasks = user_lib.cffi.NUM_NORMAL_PRODUCER
    tile_size = user_lib.cffi.TILE_SIZE
    task = user_context.create_manual_task(
        user_lib.shared_object.PRODUCE_NORMAL, Rect([num_tasks])
    )
    store = user_context.create_store(ty.int64, shape=(num_tasks * tile_size,))
    part = store.partition_by_tiling((tile_size,))
    task.add_output(part)
    task.execute()

    result = user_context.tree_reduce(
        user_lib.shared_object.REDUCE_NORMAL, store, radix=4
    )
    # The result should be a normal store
    assert not result.unbound


def test_tree_reduce_unbound():
    num_tasks = 4
    task = user_context.create_manual_task(
        user_lib.shared_object.PRODUCE_UNBOUND, Rect([num_tasks])
    )
    store = user_context.create_store(ty.int64, ndim=1)
    task.add_output(store)
    task.execute()

    result = user_context.tree_reduce(
        user_lib.shared_object.REDUCE_UNBOUND, store, radix=num_tasks
    )
    # The result should be a normal store
    assert not result.unbound


def test_tree_single_proc():
    task = user_context.create_manual_task(
        user_lib.shared_object.PRODUCE_UNBOUND, Rect([1])
    )
    store = user_context.create_store(ty.int64, ndim=1)
    task.add_output(store)
    task.execute()

    result = user_context.tree_reduce(
        user_lib.shared_object.REDUCE_UNBOUND, store, radix=4
    )
    # The result should be a normal store
    assert not result.unbound


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
