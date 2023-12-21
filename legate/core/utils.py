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

import traceback
from ctypes import CDLL, RTLD_GLOBAL
from types import TracebackType
from typing import (
    Any,
    Hashable,
    Iterable,
    Iterator,
    MutableSet,
    Optional,
    Protocol,
    TypeVar,
)

from .shape import Shape

T = TypeVar("T", bound="Hashable")


class AnyCallable(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        ...


class ShutdownCallback(Protocol):
    def __call__(self) -> None:
        ...


class OrderedSet(MutableSet[T]):
    """
    A set() variant whose iterator returns elements in insertion order.

    The implementation of this class piggybacks off of the corresponding
    iteration order guarantee for dict(), starting with Python 3.7. This is
    useful for guaranteeing symmetric execution of algorithms on different
    shards in a replicated context.
    """

    def __init__(self, copy_from: Optional[Iterable[T]] = None) -> None:
        self._dict: dict[T, None] = {}
        if copy_from is not None:
            for obj in copy_from:
                self.add(obj)

    def add(self, obj: T) -> None:
        self._dict[obj] = None

    def update(self, other: Iterable[T]) -> None:
        for obj in other:
            self.add(obj)

    def discard(self, obj: T) -> None:
        self._dict.pop(obj, None)

    def __len__(self) -> int:
        return len(self._dict)

    def __contains__(self, obj: object) -> bool:
        return obj in self._dict

    def __iter__(self) -> Iterator[T]:
        return iter(self._dict)

    def remove_all(self, other: OrderedSet[T]) -> OrderedSet[T]:
        return OrderedSet(obj for obj in self if obj not in other)


def cast_tuple(value: Any) -> tuple[Any, ...]:
    return value if isinstance(value, tuple) else tuple(value)


def capture_traceback_repr(
    skip_core_frames: bool = True,
) -> Optional[str]:
    tb = None
    for frame, _ in traceback.walk_stack(None):
        if frame.f_globals["__name__"].startswith("legate.core"):
            continue
        tb = TracebackType(
            tb,
            tb_frame=frame,
            tb_lasti=frame.f_lasti,
            tb_lineno=frame.f_lineno,
        )
    return "".join(traceback.format_tb(tb)) if tb is not None else None


def is_iterable(obj: Any) -> bool:
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def is_shape_like(obj: Any) -> bool:
    return isinstance(obj, Shape) or is_iterable(obj)


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
