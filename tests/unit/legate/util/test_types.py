# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Consolidate test configuration from command-line and environment.

"""
from __future__ import annotations

from dataclasses import dataclass

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
    quux = ["a", "b", "c"]
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
