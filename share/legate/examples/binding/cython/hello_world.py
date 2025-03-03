#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# distutils: language=c++
# cython: language_level=3

from __future__ import annotations

from hello_world_cython import HelloWorld

from legate import get_legate_runtime


def main() -> None:  # noqa: D103
    runtime = get_legate_runtime()

    lib = runtime.create_library("hello")

    hw = HelloWorld()
    hw.register_variants(lib)

    task = runtime.create_auto_task(lib, hw.TASK_ID)
    task.execute()


if __name__ == "__main__":
    main()
