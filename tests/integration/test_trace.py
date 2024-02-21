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

import pytest

from legate.core import LogicalStore, Scalar, get_legate_runtime, types as ty
from legate.core._lib.experimental.trace import Trace
from legate.core.task import InputStore, OutputStore, task


@task
def taskA(input: InputStore):
    pass


@task
def taskB(input: OutputStore, output: InputStore):
    pass


def launch_task(store: LogicalStore):
    get_legate_runtime().issue_fill(store, Scalar(42, ty.int64))
    taskA(store)
    taskB(store, store)


def test_trace():
    runtime = get_legate_runtime()
    store = runtime.create_store(ty.int64, shape=(10,))
    launch_task(store)
    for _ in range(10):
        with Trace(42):
            launch_task(store)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
