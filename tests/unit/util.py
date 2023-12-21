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

from itertools import chain, combinations
from typing import Any, Iterable, Iterator

import pytest
from typing_extensions import TypeAlias

Capsys: TypeAlias = pytest.CaptureFixture[str]


# ref: https://docs.python.org/3/library/itertools.html
def powerset(iterable: Iterable[Any]) -> Iterator[Any]:
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def powerset_nonempty(iterable: Iterable[Any]) -> Iterator[Any]:
    return (x for x in powerset(iterable) if len(x))
