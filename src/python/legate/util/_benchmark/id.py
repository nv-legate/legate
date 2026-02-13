# SPDX-FileCopyrightText: Copyright (c) 2026-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from ...core import get_legate_runtime, types as ty
from ...core.task import OutputArray, task


def _num_nodes() -> int:
    runtime = get_legate_runtime()
    machine = runtime.get_machine()
    nodes = machine.get_node_range()
    return nodes[1] - nodes[0]


@task
def _pick_one_uid(uid: np.uint64, out: OutputArray) -> None:
    np.asarray(out)[:] = uid


def _consensus_uid(uid: np.uint64) -> np.uint64:
    """Choose a consensus id from all ranks."""
    if _num_nodes() > 1:
        arr = np.array([0], dtype=np.uint64)
        store = get_legate_runtime().create_store_from_buffer(
            ty.uint64, arr.shape, arr, read_only=False
        )
        _pick_one_uid(uid, store)
        return np.asarray(store.get_physical_store())[0]
    return uid
