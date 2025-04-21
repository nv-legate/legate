# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Protocol

# imported for backwards compatibility
from ._ext.utils.ordered_set import OrderedSet

__all__ = ("Annotation", "OrderedSet")


class AnyCallable(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        pass


class ShutdownCallback(Protocol):
    def __call__(self) -> None:
        pass


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
