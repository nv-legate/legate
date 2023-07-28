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

from typing import Any

import pytest
from pytest_mock import MockerFixture
from typing_extensions import TypeAlias

import legate.util.colors as m

try:
    import colorama  # type: ignore
except ImportError:
    colorama = None

UsePlainTextFixture: TypeAlias = Any


@pytest.fixture
def use_plain_text(mocker: MockerFixture) -> None:
    mocker.patch.object(m, "bright", m._text)
    mocker.patch.object(m, "dim", m._text)
    mocker.patch.object(m, "white", m._text)
    mocker.patch.object(m, "cyan", m._text)
    mocker.patch.object(m, "red", m._text)
    mocker.patch.object(m, "green", m._text)
    mocker.patch.object(m, "yellow", m._text)
    mocker.patch.object(m, "magenta", m._text)


COLOR_FUNCS = (
    "cyan",
    "green",
    "magenta",
    "red",
    "white",
    "yellow",
)

STYLE_FUNCS = (
    "bright",
    "dim",
)


def test_default_ENABLED() -> None:
    assert m.ENABLED is False


@pytest.mark.skipif(colorama is None, reason="colorama required")
@pytest.mark.parametrize("color", COLOR_FUNCS)
def test_color_functions_ENABLED_True(
    mocker: MockerFixture, color: str
) -> None:
    mocker.patch.object(m, "ENABLED", True)

    cfunc = getattr(m, color)
    cprop = getattr(colorama.Fore, color.upper())

    out = cfunc("some text")

    assert out == f"{cprop}some text{colorama.Style.RESET_ALL}"


@pytest.mark.parametrize("color", COLOR_FUNCS)
def test_color_functions_ENABLED_False(
    mocker: MockerFixture, color: str
) -> None:
    mocker.patch.object(m, "ENABLED", False)

    cfunc = getattr(m, color)

    out = cfunc("some text")

    assert out == "some text"


@pytest.mark.skipif(colorama is None, reason="colorama required")
@pytest.mark.parametrize("style", STYLE_FUNCS)
def test_style_functions_ENABLED_True(
    mocker: MockerFixture, style: str
) -> None:
    mocker.patch.object(m, "ENABLED", True)

    sfunc = getattr(m, style)
    sprop = getattr(colorama.Style, style.upper())

    out = sfunc("some text")

    assert out == f"{sprop}some text{colorama.Style.RESET_ALL}"


@pytest.mark.parametrize("style", STYLE_FUNCS)
def test_style_functions_ENABLED_False(
    mocker: MockerFixture, style: str
) -> None:
    mocker.patch.object(m, "ENABLED", False)

    sfunc = getattr(m, style)

    out = sfunc("some text")

    assert out == "some text"


@pytest.mark.skipif(colorama is None, reason="colorama required")
@pytest.mark.parametrize("color", COLOR_FUNCS)
@pytest.mark.parametrize("style", STYLE_FUNCS)
def test_scrub(style: str, color: str) -> None:
    cfunc = getattr(m, color)
    sfunc = getattr(m, style)

    assert m.scrub(cfunc(sfunc("some text"))) == "some text"
    assert m.scrub(sfunc(cfunc("some text"))) == "some text"


@pytest.mark.skipif(colorama is None, reason="colorama required")
@pytest.mark.parametrize("color", COLOR_FUNCS)
@pytest.mark.parametrize("style", STYLE_FUNCS)
def test_scrub_plain(
    use_plain_text: UsePlainTextFixture, style: str, color: str
) -> None:
    cfunc = getattr(m, color)
    sfunc = getattr(m, style)

    assert m.scrub(cfunc(sfunc("some text"))) == "some text"
    assert m.scrub(sfunc(cfunc("some text"))) == "some text"
