# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.B
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

import pytest

from ...manager import ConfigurationManager
from .dummy_main_module import DummyMainModule

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_T = TypeVar("_T")
_P = ParamSpec("_P")


class DummyManager(ConfigurationManager):
    def log(self, *args: Any, **kwargs: Any) -> None:
        pass

    def log_divider(self, *args: Any, **kwargs: Any) -> None:
        pass

    def log_boxed(self, *args: Any, **kwargs: Any) -> None:
        pass

    def log_warning(self, *args: Any, **kwargs: Any) -> None:
        pass

    def log_execute_command(
        self, cmd: Sequence[_T], live: bool = False
    ) -> Any:
        pass

    def log_execute_func(  # type: ignore[override]
        self, func: Callable[_P, _T], *args: _P.args, **kwargs: _P.kwargs
    ) -> _T:
        return func(*args, **kwargs)


@pytest.fixture
def manager() -> DummyManager:
    return DummyManager((), DummyMainModule)
