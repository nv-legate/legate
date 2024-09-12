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

from __future__ import annotations

import traceback
from ctypes import CDLL, RTLD_GLOBAL
from types import TracebackType
from typing import Any, Protocol

# imported for backwards compatibility
from ._ext.utils.ordered_set import OrderedSet  # noqa: F401


class AnyCallable(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        pass


class ShutdownCallback(Protocol):
    def __call__(self) -> None:
        pass


def capture_traceback_repr(
    skip_legate_frames: bool = True,
) -> str | None:
    tb = None
    for frame, _ in traceback.walk_stack(None):
        if frame.f_globals["__name__"].startswith("legate"):
            continue
        tb = TracebackType(
            tb,
            tb_frame=frame,
            tb_lasti=frame.f_lasti,
            tb_lineno=frame.f_lineno,
        )
    return "".join(traceback.format_tb(tb)) if tb is not None else None


def dlopen_no_autoclose(ffi: Any, lib_path: str) -> Any:
    # Use an already-opened library handle, which cffi will convert to a
    # regular FFI object (using the definitions previously added using
    # ffi.cdef), but will not automatically dlclose() on collection.
    lib = CDLL(lib_path, mode=RTLD_GLOBAL)
    return ffi.dlopen(ffi.cast("void *", lib._handle))


class Annotation:
    def __init__(self, pairs: dict[str, str]) -> None:
        """
        Constructs a new annotation object

        Parameters
        ----------
        pairs : dict[str, str]
            Annotations as key-value pairs
        """
        # self._annotation = runtime.annotation
        self._pairs = pairs

    def __enter__(self) -> None:
        pass
        # self._annotation.update(**self._pairs)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        pass
        # for key in self._pairs.keys():
        #    self._annotation.remove(key)
