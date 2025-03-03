# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Consolidate test configuration from command-line and environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import legate.util.types as m


class TestCPUInfo:
    def test_fields(self) -> None:
        assert set(m.CPUInfo.__dataclass_fields__) == {"ids"}


class TestGPUInfo:
    def test_fields(self) -> None:
        assert set(m.GPUInfo.__dataclass_fields__) == {"id", "total"}


class Source:
    foo = 10
    bar = 10.2
    baz = "test"
    quux: ClassVar = ["a", "b", "c"]
    extra = (1, 2, 3)


@dataclass(frozen=True)
class Target(m.DataclassMixin):
    foo: int
    bar: float
    baz: str
    quux: list[str]


def test_object_to_dataclass() -> None:
    source = Source()
    target = m.object_to_dataclass(source, Target)

    assert set(target.__dict__) == set(Target.__dataclass_fields__)
    for k, v in target.__dict__.items():
        assert getattr(source, k) == v
