# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Dynamically load libnccl.so.2
#
# Based on https://github.com/rapidsai/ucx-wheels/blob/main/python/libucx/libucx/load.py

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

import os
import ctypes
from importlib.metadata import PackageNotFoundError, files
from pathlib import Path

NCCL_SONAME = "libnccl.so.2"


# Loading with RTLD_LOCAL adds the library itself to the loader's
# loaded library cache without loading any symbols into the global
# namespace. This allows libraries that express a dependency on
# a library to be loaded later and successfully satisfy that dependency
# without polluting the global symbol table with symbols from
# that library that could conflict with symbols from other DSOs.
PREFERRED_LOAD_FLAG = ctypes.RTLD_LOCAL


def _load_system_installation() -> Any:
    """Try to dlopen() the library indicated by ``soname``
    Returns ``None`` if library cannot be loaded.
    """
    try:
        return ctypes.CDLL(NCCL_SONAME, PREFERRED_LOAD_FLAG)
    except OSError:
        return None


def _load_wheel_installation() -> Any:
    """Try to dlopen() the library indicated by ``soname``
    Returns ``None`` if the library cannot be loaded.
    """
    for cuda_ver in (12, 13):
        nccl_dist = f"nvidia-nccl-cu{cuda_ver}"
        try:
            nccl_wheel_files = files(nccl_dist)
            if nccl_wheel_files is None:
                continue
            libnccl = next(
                (
                    f.locate()
                    for f in nccl_wheel_files
                    if Path(f).name == NCCL_SONAME
                ),
                None,
            )
        except PackageNotFoundError:
            continue

        if libnccl is None:
            return None

        if Path(libnccl).is_file():
            try:
                return ctypes.CDLL(str(libnccl), PREFERRED_LOAD_FLAG)
            except OSError:
                return None

    return None


def _maybe_load_library() -> Any:
    """Dynamically load libnccl."""
    prefer_system_installation = (
        os.getenv("LEGATE_NCCL_PREFER_SYSTEM_LIBRARY", "false").lower()
        != "false"
    )

    if prefer_system_installation:
        # Prefer a system library if one is present or already loaded
        lib = _load_system_installation()
        if lib is None:
            lib = _load_wheel_installation()
    else:
        # Prefer the libraries from the nvidia-nccl-cu* wheel. If they aren't
        # found (which might be the case in builds not using the CI wheel
        # scripts), look for a system installation.
        lib = _load_wheel_installation()
        if lib is None:
            lib = _load_system_installation()

    return lib


_libnccl = _maybe_load_library()
