# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import traceback
from ctypes import CDLL, RTLD_GLOBAL
from types import TracebackType
from typing import Any, Protocol

# imported for backwards compatibility
from ._ext.utils.ordered_set import OrderedSet  # noqa: F401


class AnyCallable(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: D102
        pass


class ShutdownCallback(Protocol):
    def __call__(self) -> None:  # noqa: D102
        pass


def capture_traceback_repr(  # noqa: D103
    *,
    skip_legate_frames: bool = True,  # noqa: ARG001
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


def dlopen_no_autoclose(ffi: Any, lib_path: str) -> Any:  # noqa: D103
    # Use an already-opened library handle, which cffi will convert to a
    # regular FFI object (using the definitions previously added using
    # ffi.cdef), but will not automatically dlclose() on collection.
    lib = CDLL(lib_path, mode=RTLD_GLOBAL)
    return ffi.dlopen(ffi.cast("void *", lib._handle))  # noqa: SLF001


class Annotation:
    def __init__(self, pairs: dict[str, str]) -> None:
        """
        Constructs a new annotation object.

        Parameters
        ----------
        pairs : dict[str, str]
            Annotations as key-value pairs
        """
        # self._annotation = runtime.annotation
        self._pairs = pairs

    def __enter__(self) -> None:  # noqa: D105
        pass
        # self._annotation.update(**self._pairs)

    def __exit__(  # noqa: D105
        self, _exc_type: Any, _exc_value: Any, _traceback: Any
    ) -> None:
        pass
        # for key in self._pairs.keys():
        #    self._annotation.remove(key)
