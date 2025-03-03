# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, ParamSpec, TypeVar

class Magics:
    def __init__(self, shell: Any) -> None: ...

_R = TypeVar("_R")
_P = ParamSpec("_P")

line_magic: Callable[[Callable[_P, _R]], Callable[_P, _R]]
magics_class: Callable[[Callable[_P, _R]], Callable[_P, _R]]
