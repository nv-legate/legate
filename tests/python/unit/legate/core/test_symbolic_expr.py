# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from random import randint

import pytest

from legate.core import constant, dimension


class TestConstant:
    def test_properties(self) -> None:
        val = randint(-(2 ** (32 - 1)), 2 ** (32 - 1))
        expr = constant(val)
        assert expr.__eq__(val) is NotImplemented
        assert expr.dim == -1
        assert expr.weight == 0
        assert expr.offset == val
        assert str(expr) == repr(expr)
        assert not expr.is_identity(val)

    @pytest.mark.parametrize("val", [randint(-100, 100) for _ in range(2)])
    @pytest.mark.parametrize("other", [randint(2, 100) for _ in range(2)])
    def test_operators(self, val: int, other: int) -> None:
        expr = constant(val)
        add_result = expr + other
        mul_result = expr * other
        assert add_result.offset == val + other
        assert mul_result.offset == val * other
        assert expr != add_result != mul_result


class TestDimension:
    def test_properties(self) -> None:
        val = randint(-(2 ** (32 - 1)), 2 ** (32 - 1))
        expr = dimension(val)
        assert expr.dim == val
        assert expr.weight == 1
        assert expr.offset == 0
        assert str(expr) == repr(expr)
        assert expr.is_identity(val)
        assert not expr.is_identity(val - 1)

    @pytest.mark.parametrize("val", [randint(-100, 100) for _ in range(2)])
    @pytest.mark.parametrize("other", [randint(-100, 100) for _ in range(2)])
    def test_operators(self, val: int, other: int) -> None:
        expr = dimension(val)
        assert expr.__eq__(val) is NotImplemented
        add_result = expr + other
        mul_result = expr * other
        assert add_result.offset == other
        assert mul_result.weight == other
        assert expr != add_result != mul_result
