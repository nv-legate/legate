#!/usr/bin/env python3
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
# distutils: language=c++
# cython: language_level=3

from __future__ import annotations

from hello_world_cython import HelloWorld

from legate.core import get_legate_runtime


def main() -> None:
    runtime = get_legate_runtime()

    lib = runtime.create_library("hello")

    hw = HelloWorld()
    hw.register_variants(lib)

    task = runtime.create_auto_task(lib, hw.TASK_ID)
    task.execute()


if __name__ == "__main__":
    main()
