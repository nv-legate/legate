# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from .. import install_info as info

__all__ = ("print_build_info",)


def print_build_info() -> None:  # noqa: D103
    print(  # noqa: T201
        f"""Legate build configuration:
  build_type : {info.build_type}
  use_openmp : {info.use_openmp}
  use_cuda   : {info.use_cuda}
  networks   : {",".join(info.networks) if info.networks else ""}
  conduit    : {info.conduit}
"""
    )
