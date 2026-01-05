# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

class Magics:
    def __init__(self, shell: Any) -> None: ...

_R = TypeVar("_R")
_P = ParamSpec("_P")

def line_magic(f: Callable[_P, _R]) -> Callable[_P, _R]: ...
def magics_class(f: Callable[_P, _R]) -> Callable[_P, _R]: ...
