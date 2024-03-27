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
from argparse import ArgumentParser
from typing import Any

import pytest

from ...cmake.cmake_flags import CMAKE_VARIABLE, CMakeString, _CMakeVar
from ...util.argument_parser import (
    ArgSpec,
    ConfigArgument,
    ExclusiveArgumentGroup,
    Unset,
    _str_to_bool,
)
from ...util.exception import LengthError


class TestConfigArgument:
    @pytest.mark.parametrize(
        "cmake_var", (None, CMAKE_VARIABLE("FOO", CMakeString))
    )
    @pytest.mark.parametrize("ephemeral", (True, False))
    def test_create(
        self, cmake_var: _CMakeVar | None, ephemeral: bool
    ) -> None:
        arg = ConfigArgument(
            name="--foo",
            spec=ArgSpec(dest="bar"),
            cmake_var=cmake_var,
            ephemeral=ephemeral,
        )
        assert arg.name == "--foo"
        assert arg.spec == ArgSpec(dest="bar")
        assert arg.cmake_var == cmake_var
        assert arg.ephemeral == ephemeral

    @pytest.mark.parametrize(
        "cmake_var", (None, CMAKE_VARIABLE("FOO", CMakeString))
    )
    @pytest.mark.parametrize("ephemeral", (True, False))
    @pytest.mark.parametrize("ty", (None, int, str, bool))
    @pytest.mark.parametrize(
        "const,nargs", ((Unset, Unset), (True, "?"), (False, "?"))
    )
    @pytest.mark.parametrize("default", (Unset, True, False))
    @pytest.mark.parametrize("metavar", (Unset, "foo"))
    def test_add_to_argparser(
        self,
        cmake_var: _CMakeVar | None,
        ephemeral: bool,
        ty: type,
        nargs: str | Any,
        const: bool | Any,
        default: bool | Any,
        metavar: str | Any,
    ) -> None:
        parser = ArgumentParser()
        arg = ConfigArgument(
            name="--foo",
            spec=ArgSpec(
                dest="bar",
                type=ty,
                nargs=nargs,  # type: ignore[arg-type]
                const=const,
                default=default,
                metavar=metavar,
            ),
            cmake_var=cmake_var,
            ephemeral=ephemeral,
        )

        arg.add_to_argparser(parser)
        assert len(parser._actions) == 2  # has implicit help action
        action = parser._actions[1]
        assert action.option_strings == ["--foo"]
        assert action.dest == "bar"
        if ty is bool:
            assert action.nargs == ("?" if nargs is Unset else nargs)
            assert action.const is (True if const is Unset else const)
            assert action.default is (False if default is Unset else default)
            assert action.metavar == ("bool" if metavar is Unset else metavar)
            assert action.type is _str_to_bool
        else:
            assert action.nargs == (None if nargs is Unset else nargs)
            assert action.const == (None if const is Unset else const)
            assert action.default == (None if default is Unset else default)
            assert action.metavar == (None if metavar is Unset else metavar)
            assert action.type is ty  # type: ignore[comparison-overlap]
        assert action.choices is None
        assert action.required is False
        assert action.help is None


class TestExclusiveArgumentGroup:
    @pytest.mark.parametrize("required", (True, False))
    def test_create(self, required: bool) -> None:
        foo = ConfigArgument(name="--foo", spec=ArgSpec(dest="foo"))
        bar = ConfigArgument(name="--bar", spec=ArgSpec(dest="bar"))
        group = ExclusiveArgumentGroup(required=required, Foo=foo, Bar=bar)

        assert group.group == {"Foo": foo, "Bar": bar}
        assert group.Foo is foo  # type: ignore[attr-defined]
        assert group.Bar is bar  # type: ignore[attr-defined]
        assert group.required == required

    def test_create_bad(self) -> None:
        foo = ConfigArgument(name="--foo", spec=ArgSpec(dest="foo"))
        with pytest.raises(
            LengthError,
            match="Must supply at least 2 arguments to exclusive group",
        ):
            ExclusiveArgumentGroup()

        with pytest.raises(
            LengthError,
            match="Must supply at least 2 arguments to exclusive group",
        ):
            ExclusiveArgumentGroup(foo=foo)

        bar = 10
        with pytest.raises(
            TypeError, match=f"Argument Bar wrong type: {type(bar)}"
        ):
            ExclusiveArgumentGroup(Foo=foo, Bar=bar)  # type: ignore[arg-type]


if __name__ == "__main__":
    sys.exit(pytest.main())
