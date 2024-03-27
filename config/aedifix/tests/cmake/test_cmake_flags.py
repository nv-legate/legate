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

import copy
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

from ...cmake.cmake_flags import (
    CMakeBool,
    CMakeExecutable,
    CMakeInt,
    CMakeList,
    CMakePath,
    CMakeString,
)


class TestCMakeList:
    def test_create(self) -> None:
        var = CMakeList("foo")
        assert var.name == "foo"
        assert var.value is None
        assert var.prefix == "-D"
        assert var.type == "STRING"

        var = CMakeList("foo", value=[1, 2, 3])
        assert var.name == "foo"
        assert var.value == [1, 2, 3]
        assert var.prefix == "-D"
        assert var.type == "STRING"

        var = CMakeList("foo", value=(1, 2, 3), prefix="bar")
        assert var.name == "foo"
        assert var.value == [1, 2, 3]
        assert var.prefix == "bar"
        assert var.type == "STRING"

    def test_create_bad(self) -> None:
        with pytest.raises(TypeError):
            CMakeList("foo", value="hello")

    def test_canonicalize(self) -> None:
        var = CMakeList("foo")
        canon = var.canonicalize()
        assert canon is None
        assert id(canon) != id(var)  # must be distinct
        assert var.name == "foo"
        assert var.value is None
        assert var.prefix == "-D"
        assert var.type == "STRING"

        var.value = [4, 5, 6]
        canon = var.canonicalize()
        assert isinstance(canon, CMakeList)
        assert id(canon) != id(var)  # must be distinct
        assert canon.name == var.name
        assert canon.value == list(map(str, var.value))
        assert id(canon.value) != id(var.value)  # must be distinct
        assert canon.prefix == var.prefix
        assert canon.type == var.type

    @pytest.mark.parametrize(
        "value", (None, [], ["1"], ["1", "2"], [34, 99, 999])
    )
    def test_to_command_line(self, value: list[str]) -> None:
        val_copy = copy.deepcopy(value)
        var = CMakeList("foo", value=value)
        cmd = var.to_command_line()
        assert isinstance(cmd, str)
        expected_str = "" if val_copy is None else " ".join(map(str, val_copy))
        assert cmd == f"-Dfoo:STRING={expected_str}"
        assert var.value == val_copy

    def test_eq(self) -> None:
        lhs = CMakeList("foo", value=(1, 2, 3), prefix="bar")
        rhs = CMakeList("foo", value=(1, 2, 3), prefix="bar")
        assert lhs == rhs

    def test_neq(self) -> None:
        lhs = CMakeList("foo", value=(1, 2, 3), prefix="bar")

        rhs = CMakeList("bar", value=(1, 2, 3), prefix="bar")
        assert lhs != rhs

        rhs = CMakeList("foo", value=(1, 2, 3, 4), prefix="bar")
        assert lhs != rhs

        rhs = CMakeList("foo", value=(1, 2, 3), prefix="asdasd")
        assert lhs != rhs

        rhs_b = CMakeBool("foo", value=None, prefix="asdasd")
        assert lhs != rhs_b


class TestCMakeBool:
    def test_create(self) -> None:
        var = CMakeBool("foo")
        assert var.name == "foo"
        assert var.value is None
        assert var.prefix == "-D"
        assert var.type == "BOOL"

        var = CMakeBool("foo", value=True)
        assert var.name == "foo"
        assert var.value == "ON"
        assert var.prefix == "-D"
        assert var.type == "BOOL"

        var = CMakeBool("foo", value=1, prefix="bar")
        assert var.name == "foo"
        assert var.value == "ON"
        assert var.prefix == "bar"
        assert var.type == "BOOL"

        var = CMakeBool("foo", value=False, prefix="bar")
        assert var.name == "foo"
        assert var.value == "OFF"
        assert var.prefix == "bar"
        assert var.type == "BOOL"

    def test_create_bad(self) -> None:
        with pytest.raises(ValueError):
            CMakeBool("foo", value="hello")

        with pytest.raises(ValueError):
            CMakeBool("foo", value=400)

        with pytest.raises(ValueError):
            CMakeBool("foo", value="off")

        with pytest.raises(TypeError):
            CMakeBool("foo", value=1.0)  # type: ignore[arg-type]

    def test_canonicalize(self) -> None:
        var = CMakeBool("foo")
        canon = var.canonicalize()
        assert canon is None
        assert id(canon) != id(var)  # must be distinct
        assert var.name == "foo"
        assert var.value is None
        assert var.prefix == "-D"
        assert var.type == "BOOL"

        var.value = True
        canon = var.canonicalize()
        assert isinstance(canon, CMakeBool)
        assert id(canon) != id(var)  # must be distinct
        assert canon.name == var.name
        assert canon.value == "ON"
        assert canon.prefix == var.prefix
        assert canon.type == var.type

    @pytest.mark.parametrize("value", (True, False, 0, 1))
    def test_to_command_line(self, value: Any) -> None:
        val_copy = copy.deepcopy(value)
        var = CMakeBool("foo", value=value)
        cmd = var.to_command_line()
        assert isinstance(cmd, str)
        expected_str = (
            "" if val_copy is None else ("ON" if val_copy else "OFF")
        )
        assert cmd == f"-Dfoo:BOOL={expected_str}"
        assert var.value == expected_str

    def test_to_command_line_bad(self) -> None:
        var = CMakeBool("foo", value=None)
        with pytest.raises(
            ValueError,
            match='Cannot convert "foo" to command-line, have empty value',
        ):
            var.to_command_line()

    def test_eq(self) -> None:
        lhs = CMakeBool("foo", value=True, prefix="bar")
        rhs = CMakeBool("foo", value=True, prefix="bar")
        assert lhs == rhs

    def test_neq(self) -> None:
        lhs = CMakeBool("foo", value=True, prefix="bar")

        rhs = CMakeBool("bar", value=True, prefix="bar")
        assert lhs != rhs

        rhs = CMakeBool("foo", value=False, prefix="bar")
        assert lhs != rhs

        rhs = CMakeBool("foo", value=True, prefix="asdasd")
        assert lhs != rhs

        rhs_i = CMakeInt("foo", value=1, prefix="bar")
        assert lhs != rhs_i


class TestCMakeInt:
    def test_create(self) -> None:
        var = CMakeInt("foo")
        assert var.name == "foo"
        assert var.value is None
        assert var.prefix == "-D"
        assert var.type == "STRING"

        var = CMakeInt("foo", value=0)
        assert var.name == "foo"
        assert var.value == 0
        assert var.prefix == "-D"
        assert var.type == "STRING"

        var = CMakeInt("foo", value=10, prefix="bar")
        assert var.name == "foo"
        assert var.value == 10
        assert var.prefix == "bar"
        assert var.type == "STRING"

        var = CMakeInt("foo", value=10.0)
        assert var.name == "foo"
        assert var.value == 10
        assert var.prefix == "-D"
        assert var.type == "STRING"

        var = CMakeInt("foo", value=True)
        assert var.name == "foo"
        assert var.value == 1
        assert var.prefix == "-D"
        assert var.type == "STRING"

        var = CMakeInt("foo", value="23")
        assert var.name == "foo"
        assert var.value == 23
        assert var.prefix == "-D"
        assert var.type == "STRING"

    def test_create_bad(self) -> None:
        with pytest.raises(ValueError):
            CMakeInt("foo", value="hello")

        with pytest.raises(TypeError):
            CMakeInt("foo", value=complex(1, 2))  # type: ignore[arg-type]

    def test_canonicalize(self) -> None:
        var = CMakeInt("foo")
        canon = var.canonicalize()
        assert canon is None
        assert id(canon) != id(var)  # must be distinct
        assert var.name == "foo"
        assert var.value is None
        assert var.prefix == "-D"
        assert var.type == "STRING"

        var.value = 45
        canon = var.canonicalize()
        assert isinstance(canon, CMakeInt)
        assert id(canon) != id(var)  # must be distinct
        assert canon.name == var.name
        assert canon.value == var.value
        assert canon.prefix == var.prefix
        assert canon.type == var.type

    @pytest.mark.parametrize("value", (0, 1, 10, True, False, 123.45, "38"))
    def test_to_command_line(self, value: Any) -> None:
        val_copy = copy.deepcopy(int(value))
        var = CMakeInt("foo", value=value)
        cmd = var.to_command_line()
        assert isinstance(cmd, str)
        assert cmd == f"-Dfoo:STRING={val_copy}"
        assert var.value == val_copy

    def test_to_command_line_bad(self) -> None:
        var = CMakeInt("foo", value=None)
        with pytest.raises(
            ValueError,
            match='Cannot convert "foo" to command-line, have empty value',
        ):
            var.to_command_line()

    def test_eq(self) -> None:
        lhs = CMakeInt("foo", value=12, prefix="bar")
        rhs = CMakeInt("foo", value=12, prefix="bar")
        assert lhs == rhs

    def test_neq(self) -> None:
        lhs = CMakeInt("foo", value=45, prefix="bar")

        rhs = CMakeInt("asdasd", value=45, prefix="bar")
        assert lhs != rhs

        rhs = CMakeInt("foo", value=12, prefix="bar")
        assert lhs != rhs

        rhs = CMakeInt("foo", value=45, prefix="asdasd")
        assert lhs != rhs

        rhs_s = CMakeString("foo", value="", prefix="bar")
        assert lhs != rhs_s


class TestCMakeString:
    def test_create(self) -> None:
        var = CMakeString("foo")
        assert var.name == "foo"
        assert var.value is None
        assert var.prefix == "-D"
        assert var.type == "STRING"

        var = CMakeString("foo", value="0")
        assert var.name == "foo"
        assert var.value == "0"
        assert var.prefix == "-D"
        assert var.type == "STRING"

        var = CMakeString("foo", value="asdasd", prefix="bar")
        assert var.name == "foo"
        assert var.value == "asdasd"
        assert var.prefix == "bar"
        assert var.type == "STRING"

    def test_create_bad(self) -> None:
        with pytest.raises(TypeError):
            CMakeString("foo", value=1)  # type: ignore[arg-type]

        with pytest.raises(TypeError):
            CMakeString("foo", value=complex(1, 2))  # type: ignore[arg-type]

        with pytest.raises(TypeError):
            CMakeString("foo", value=[1, 2])  # type: ignore[arg-type]

    def test_canonicalize(self) -> None:
        var = CMakeString("foo")
        canon = var.canonicalize()
        assert canon is None
        assert id(canon) != id(var)  # must be distinct
        assert var.name == "foo"
        assert var.value is None
        assert var.prefix == "-D"
        assert var.type == "STRING"

        var.value = "hello world"
        canon = var.canonicalize()
        assert isinstance(canon, CMakeString)
        assert id(canon) != id(var)  # must be distinct
        assert canon.name == var.name
        assert canon.value == var.value
        assert canon.prefix == var.prefix
        assert canon.type == var.type

    @pytest.mark.parametrize("value", ("hello", "goodbye", "38"))
    def test_to_command_line(self, value: str) -> None:
        val_copy = copy.deepcopy(value)
        var = CMakeString("foo", value=value)
        cmd = var.to_command_line()
        assert isinstance(cmd, str)
        assert cmd == f"-Dfoo:STRING={val_copy}"
        assert var.value == val_copy

    def test_to_command_line_bad(self) -> None:
        var = CMakeString("foo", value=None)
        with pytest.raises(
            ValueError,
            match='Cannot convert "foo" to command-line, have empty value',
        ):
            var.to_command_line()

    def test_eq(self) -> None:
        lhs = CMakeString("foo", value="hello", prefix="bar")
        rhs = CMakeString("foo", value="hello", prefix="bar")
        assert lhs == rhs

    def test_neq(self) -> None:
        lhs = CMakeString("foo", value="hello", prefix="bar")

        rhs = CMakeString("asdads", value="hello", prefix="bar")
        assert lhs != rhs

        rhs = CMakeString("foo", value="asdasd", prefix="bar")
        assert lhs != rhs

        rhs = CMakeString("foo", value="hello", prefix="asdas")
        assert lhs != rhs

        rhs_p = CMakePath("foo", value="/foo/bar", prefix="bar")
        assert lhs != rhs_p


class TestCMakePath:
    def test_create(self) -> None:
        var = CMakePath("foo")
        assert var.name == "foo"
        assert var.value is None
        assert var.prefix == "-D"
        assert var.type == "STRING"

        var = CMakePath("foo", value="/foo/bar/baz")
        assert var.name == "foo"
        assert var.value == Path("/foo/bar/baz").resolve()
        assert var.prefix == "-D"
        assert var.type == "STRING"

        var = CMakePath("foo", value=Path("/foo/bar/baz"), prefix="bar")
        assert var.name == "foo"
        assert var.value == Path("/foo/bar/baz").resolve()
        assert var.prefix == "bar"
        assert var.type == "STRING"

        var = CMakePath("foo", value=__file__)
        assert var.name == "foo"
        assert var.value == Path(__file__).resolve()
        assert var.prefix == "-D"
        assert var.type == "FILEPATH"

        var = CMakePath("foo", value=Path(__file__))
        assert var.name == "foo"
        assert var.value == Path(__file__).resolve()
        assert var.prefix == "-D"
        assert var.type == "FILEPATH"

        var = CMakePath("foo", value=Path(__file__).parent)
        assert var.name == "foo"
        assert var.value == Path(__file__).resolve().parent
        assert var.prefix == "-D"
        assert var.type == "PATH"

    def test_create_bad(self) -> None:
        with pytest.raises(TypeError):
            CMakePath("foo", value=1)  # type: ignore[arg-type]

        with pytest.raises(TypeError):
            CMakePath("foo", value=complex(1, 2))  # type: ignore[arg-type]

        with pytest.raises(TypeError):
            CMakePath("foo", value=[1, 2])  # type: ignore[arg-type]

    def test_canonicalize(self) -> None:
        var = CMakePath("foo")
        canon = var.canonicalize()
        assert canon is None
        assert id(canon) != id(var)  # must be distinct
        assert var.name == "foo"
        assert var.value is None
        assert var.prefix == "-D"
        assert var.type == "STRING"

        var.value = Path(__file__)
        canon = var.canonicalize()
        assert isinstance(canon, CMakePath)
        assert id(canon) != id(var)  # must be distinct
        assert canon.name == var.name
        assert canon.value == var.value
        assert canon.prefix == var.prefix
        assert canon.type == var.type

    @pytest.mark.parametrize(
        "value", ("/hello/world", "/goodbye/world", __file__)
    )
    def test_to_command_line(self, value: str) -> None:
        val_copy = copy.deepcopy(Path(value).resolve())
        var = CMakePath("foo", value=value)
        cmd = var.to_command_line()
        assert isinstance(cmd, str)
        if val_copy.exists():
            type_str = "PATH" if val_copy.is_dir() else "FILEPATH"
        else:
            type_str = "STRING"
        assert cmd == f"-Dfoo:{type_str}={val_copy}"
        assert var.value == val_copy

    def test_to_command_line_bad(self) -> None:
        var = CMakePath("foo", value=None)
        with pytest.raises(
            ValueError,
            match='Cannot convert "foo" to command-line, have empty value',
        ):
            var.to_command_line()

    def test_eq(self) -> None:
        lhs = CMakePath("foo", value="/foo/bar", prefix="bar")
        rhs = CMakePath("foo", value="/foo/bar", prefix="bar")
        assert lhs == rhs

    def test_neq(self) -> None:
        lhs = CMakePath("foo", value="/foo/bar", prefix="bar")

        rhs = CMakePath("asdasd", value="/foo/bar", prefix="bar")
        assert lhs != rhs

        rhs = CMakePath("foo", value="/foo/bar/baz", prefix="bar")
        assert lhs != rhs

        rhs = CMakePath("foo", value="/foo/bar", prefix="asdasd")
        assert lhs != rhs

        rhs_e = CMakeExecutable("foo", value="/foo/bar", prefix="bar")
        assert lhs != rhs_e


class TestCMakeExecutable:
    def test_create(self) -> None:
        var = CMakeExecutable("foo")
        assert var.name == "foo"
        assert var.value is None
        assert var.prefix == "-D"
        assert var.type == "FILEPATH"

        var = CMakeExecutable("foo", value="/foo/bar/baz")
        assert var.name == "foo"
        assert var.value == Path("/foo/bar/baz").resolve()
        assert var.prefix == "-D"
        assert var.type == "FILEPATH"

        var = CMakeExecutable("foo", value=Path("/foo/bar/baz"), prefix="bar")
        assert var.name == "foo"
        assert var.value == Path("/foo/bar/baz").resolve()
        assert var.prefix == "bar"
        assert var.type == "FILEPATH"

        var = CMakeExecutable("foo", value=sys.executable)
        assert var.name == "foo"
        assert var.value == Path(sys.executable)
        assert var.prefix == "-D"
        assert var.type == "FILEPATH"

    def test_create_bad(self) -> None:
        with pytest.raises(
            ValueError, match="Got a directory as an executable: .*"
        ):
            CMakeExecutable("foo", value=Path(sys.executable).parent)

        with pytest.raises(TypeError):
            CMakeExecutable("foo", value=1)  # type: ignore[arg-type]

        with pytest.raises(TypeError):
            CMakeExecutable(
                "foo", value=complex(1, 2)  # type: ignore[arg-type]
            )

        with pytest.raises(TypeError):
            CMakeExecutable("foo", value=[1, 2])  # type: ignore[arg-type]

    def test_canonicalize(self) -> None:
        var = CMakeExecutable("foo")
        canon = var.canonicalize()
        assert canon is None
        assert id(canon) != id(var)  # must be distinct
        assert var.name == "foo"
        assert var.value is None
        assert var.prefix == "-D"
        assert var.type == "FILEPATH"

        var.value = Path(__file__)
        canon = var.canonicalize()
        assert isinstance(canon, CMakeExecutable)
        assert id(canon) != id(var)  # must be distinct
        assert canon.name == var.name
        assert canon.value == var.value
        assert canon.prefix == var.prefix
        assert canon.type == var.type

    @pytest.mark.parametrize(
        "value", (shutil.which("ls"), shutil.which("gcc"), __file__)
    )
    def test_to_command_line(self, value: str | None) -> None:
        if value is None:
            return  # test is not meaningful if these are not found

        val_copy = copy.deepcopy(Path(value))
        var = CMakeExecutable("foo", value=value)
        cmd = var.to_command_line()
        assert isinstance(cmd, str)
        assert cmd == f"-Dfoo:FILEPATH={val_copy}"
        assert var.value == val_copy

    def test_to_command_line_bad(self) -> None:
        var = CMakeExecutable("foo", value=None)
        with pytest.raises(
            ValueError,
            match='Cannot convert "foo" to command-line, have empty value',
        ):
            var.to_command_line()

    def test_eq(self) -> None:
        lhs = CMakeExecutable("foo", value="/foo/bar/baz.py", prefix="bar")
        rhs = CMakeExecutable("foo", value="/foo/bar/baz.py", prefix="bar")
        assert lhs == rhs

    def test_neq(self) -> None:
        lhs = CMakeExecutable("foo", value="/foo/bar/baz.py", prefix="bar")

        rhs = CMakeExecutable("asdasd", value="/foo/bar/baz.py", prefix="bar")
        assert lhs != rhs

        rhs = CMakeExecutable("foo", value="/foo/bar/bop.py", prefix="bar")
        assert lhs != rhs

        rhs = CMakeExecutable("foo", value="/foo/bar/baz.py", prefix="asdasd")
        assert lhs != rhs

        rhs_l = CMakeList("foo", value=(1, 2), prefix="bar")
        assert lhs != rhs_l


if __name__ == "__main__":
    sys.exit(pytest.main())
