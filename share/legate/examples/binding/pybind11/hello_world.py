# noqa: INP001
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from hello_world_pybind11 import HelloWorld

from legate import get_legate_runtime


def main() -> None:
    """."""
    runtime = get_legate_runtime()
    lib = runtime.create_library("hello")

    HelloWorld().register_variants(lib.raw_handle)

    task = runtime.create_auto_task(lib, HelloWorld.TASK_ID)
    runtime.submit(task)


if __name__ == "__main__":
    main()
