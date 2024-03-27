#!/usr/bin/env python3
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

import sys

import pytest

from ...cmake.cmake_flags import CMakeBool, CMakeInt, CMakeList, CMakeString
from ...cmake.cmaker import CMaker
from ...util.exception import WrongOrderError
from ..fixtures.dummy_manager import DummyManager


@pytest.fixture
def cmaker() -> CMaker:
    return CMaker()


class TestCMaker:
    def test_create(self) -> None:
        cmaker = CMaker()
        assert cmaker._args == {}

    def test_register_variable(
        self, cmaker: CMaker, manager: DummyManager
    ) -> None:
        var = CMakeString("foo", value="bar")
        cmaker.register_variable(manager, var)
        assert var.name in cmaker._args
        assert cmaker._args[var.name] == var

        var2 = CMakeInt("bar", value=2)
        cmaker.register_variable(manager, var2)
        assert var.name in cmaker._args
        assert var2.name in cmaker._args
        assert cmaker._args[var.name] == var
        assert cmaker._args[var2.name] == var2

    def test_register_bad(self, cmaker: CMaker, manager: DummyManager) -> None:
        var = CMakeString("foo", value="bar")
        cmaker.register_variable(manager, var)
        assert var.name in cmaker._args
        assert cmaker._args[var.name] == var

        # same name, different kind
        var2 = CMakeInt("foo", value=2)
        with pytest.raises(
            ValueError,
            match=(
                f"Variable foo already registered as kind {type(var)}, "
                "cannot overwrite it!"
            ),
        ):
            cmaker.register_variable(manager, var2)

    def test_set_value(self, cmaker: CMaker, manager: DummyManager) -> None:
        var = CMakeBool("foo")
        assert var.value is None
        cmaker.register_variable(manager, var)
        assert cmaker._args[var.name] == var
        cmaker.set_value(manager, "foo", False)
        assert cmaker._args[var.name] == var
        assert cmaker._args[var.name].value == "OFF"

    def test_set_value_bad(
        self, cmaker: CMaker, manager: DummyManager
    ) -> None:
        with pytest.raises(
            WrongOrderError,
            match="No variable with name 'foo' has been registered",
        ):
            cmaker.set_value(manager, "foo", 1234)

    def test_get_value(self, cmaker: CMaker, manager: DummyManager) -> None:
        var = CMakeBool("foo")
        assert var.value is None
        cmaker.register_variable(manager, var)
        assert cmaker._args[var.name] == var
        value = cmaker.get_value(manager, "foo")
        assert value is None
        assert var.value is None

        var.value = True
        value = cmaker.get_value(manager, "foo")
        assert value == "ON"

    def test_get_value_bad(
        self, cmaker: CMaker, manager: DummyManager
    ) -> None:
        with pytest.raises(
            WrongOrderError,
            match="No variable with name 'foo' has been registered",
        ):
            cmaker.get_value(manager, "foo")

    def test_append_value(self, cmaker: CMaker, manager: DummyManager) -> None:
        var = CMakeList("foo")
        assert var.value is None
        cmaker.register_variable(manager, var)

        cmaker.append_value(manager, "foo", [1, 2, 3])
        assert var.value == [1, 2, 3]
        assert cmaker._args[var.name].value == [1, 2, 3]

        # no change
        cmaker.append_value(manager, "foo", [])
        assert var.value == [1, 2, 3]
        assert cmaker._args[var.name].value == [1, 2, 3]

        cmaker.append_value(manager, "foo", [4, 5, 6])
        assert var.value == [1, 2, 3, 4, 5, 6]
        assert cmaker._args[var.name].value == [1, 2, 3, 4, 5, 6]

        cmaker.append_value(manager, "foo", ["7", "8"])
        assert var.value == [1, 2, 3, 4, 5, 6, "7", "8"]
        assert cmaker._args[var.name].value == [1, 2, 3, 4, 5, 6, "7", "8"]

    def test_append_value_bad(
        self, cmaker: CMaker, manager: DummyManager
    ) -> None:
        with pytest.raises(
            WrongOrderError,
            match="No variable with name 'foo' has been registered",
        ):
            cmaker.append_value(manager, "foo", [1, 2])

        var_int = CMakeInt("foo")
        cmaker.register_variable(manager, var_int)

        with pytest.raises(
            TypeError,
            match=f"Cannot append to {type(var_int)}",
        ):
            cmaker.append_value(manager, "foo", [1, 2])


if __name__ == "__main__":
    sys.exit(pytest.main())
