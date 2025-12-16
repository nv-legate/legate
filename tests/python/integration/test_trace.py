# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os

import pytest

from legate.core import LogicalStore, Scalar, get_legate_runtime, types as ty
from legate.core._lib.experimental.trace import Trace
from legate.core.task import InputStore, OutputStore, task


@task
def taskA(store: InputStore) -> None:
    pass


@task
def taskB(ostore: OutputStore, ostore2: InputStore) -> None:
    pass


def launch_task(store: LogicalStore) -> None:
    get_legate_runtime().issue_fill(store, Scalar(42, ty.int64))
    taskA(store)
    taskB(store, store)


@pytest.mark.skipif(
    not bool(os.getenv("COVERAGE_RUN")),
    reason="Tracing cannot be used until we add "
    "launch-time return size handling",
)
def test_trace() -> None:
    runtime = get_legate_runtime()
    store = runtime.create_store(ty.int64, shape=(10,))
    launch_task(store)
    for _ in range(10):
        with Trace(42):
            launch_task(store)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
