# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from itertools import chain, combinations
from typing import TYPE_CHECKING, Any, TypeAlias

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

Capsys: TypeAlias = pytest.CaptureFixture[str]


# ref: https://docs.python.org/3/library/itertools.html
def powerset(iterable: Iterable[Any]) -> Iterator[Any]:
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def powerset_nonempty(iterable: Iterable[Any]) -> Iterator[Any]:
    return (x for x in powerset(iterable) if len(x))
