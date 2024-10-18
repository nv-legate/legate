# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

from datetime import timedelta
from shlex import quote
from typing import Any

import pytest

from legate.util import ui as m


def test_UI_WIDTH() -> None:
    assert m.UI_WIDTH == 80


def test_banner_simple() -> None:
    assert (
        m.banner("some text")
        == "\n" + "#" * m.UI_WIDTH + "\n### some text\n" + "#" * m.UI_WIDTH
    )


def test_banner_full() -> None:
    assert (
        m.banner("some text", char="*", width=100, details=["a", "b"])
        == "\n"
        + "*" * 100
        + "\n*** \n*** some text\n*** \n*** a\n*** b\n*** \n"
        + "*" * 100
    )


def test_error() -> None:
    assert m.error("some message") == "[red]ERROR: some message[/]"


def test_key() -> None:
    assert m.key("some key") == "[dim green]some key[/]"


def test_value() -> None:
    assert m.value("some value") == "[yellow]some value[/]"


class Test_kvtable:
    ONE = {"foo": 10}
    TWO = {"foo": 10, "barbaz": "some value"}
    THREE = {"foo": 10, "barbaz": "some value", "a": 1.2}

    @pytest.mark.parametrize("items", (ONE, TWO, THREE))
    def test_default(self, items: dict[str, Any]) -> None:
        N = max(len(m.key(k)) for k in items)
        assert m.kvtable(items) == "\n".join(
            f"{m.key(k): <{N}} : {m.value(quote(str(items[k])))}"
            for k in items
        )

    @pytest.mark.parametrize("items", (ONE, TWO, THREE))
    def test_delim(self, items: dict[str, Any]) -> None:
        N = max(len(m.key(k)) for k in items)
        assert m.kvtable(items, delim="/") == "\n".join(
            f"{m.key(k): <{N}}/{m.value(quote(str(items[k])))}" for k in items
        )

    @pytest.mark.parametrize("items", (ONE, TWO, THREE))
    def test_align_False(self, items: dict[str, Any]) -> None:
        assert m.kvtable(items, align=False) == "\n".join(
            f"{m.key(k)} : {m.value(quote(str(items[k])))}" for k in items
        )

    def test_keys(self) -> None:
        items = self.THREE
        keys = ("foo", "a")
        N = max(len(m.key(k)) for k in items)

        assert m.kvtable(self.THREE, keys=keys) == "\n".join(
            f"{m.key(k): <{N}} : {m.value(str(items[k]))}" for k in keys
        )


class Test_rule:
    def test_pad(self) -> None:
        assert m.rule(pad=4) == "[cyan]    " + "-" * (m.UI_WIDTH - 4) + "[/]"

    def test_pad_with_text(
        self,
    ) -> None:
        front = "    --- foo bar "
        assert (
            m.rule("foo bar", pad=4)
            == "[cyan]" + front + "-" * (m.UI_WIDTH - len(front)) + "[/]"
        )

    def test_text(self) -> None:
        front = "--- foo bar "
        assert (
            m.rule("foo bar")
            == "[cyan]" + front + "-" * (m.UI_WIDTH - len(front)) + "[/]"
        )

    def test_char(self) -> None:
        assert m.rule(char="a") == "[cyan]" + "a" * m.UI_WIDTH + "[/]"

    def test_N(self) -> None:
        assert m.rule(N=60) == "[cyan]" + "-" * 60 + "[/]"

    def test_N_with_text(self) -> None:
        front = "--- foo bar "
        assert (
            m.rule("foo bar", N=65)
            == "[cyan]" + front + "-" * (65 - len(front)) + "[/]"
        )


def test_section() -> None:
    assert m.section("some section") == "[bright white]some section[/]"


def test_warn() -> None:
    assert m.warn("some message") == "[magenta]WARNING: some message[/]"


def test_shell() -> None:
    assert m.shell("cmd --foo") == "[dim white]+cmd --foo[/]"


def test_shell_with_char() -> None:
    assert m.shell("cmd --foo", char="") == "[dim white]cmd --foo[/]"


def test_passed() -> None:
    assert m.passed("msg") == "[bold green][PASS][/] msg"


def test_passed_with_details() -> None:
    assert (
        m.passed("msg", details=["a", "b"])
        == "[bold green][PASS][/] msg\n   a\n   b"
    )


def test_failed() -> None:
    assert m.failed("msg") == "[bold red][FAIL][/] msg"


def test_failed_with_exit_code() -> None:
    fail = "[bold red][FAIL][/]"
    exit = " [bold white](exit: 10)[/]"
    assert m.failed("msg", exit_code=10) == f"{fail} msg{exit}"  # noqa


def test_failed_with_details() -> None:
    assert (
        m.failed("msg", details=["a", "b"])
        == "[bold red][FAIL][/] msg\n   a\n   b"
    )


def test_failed_with_details_and_exit_code() -> None:
    fail = "[bold red][FAIL][/]"
    exit = " [bold white](exit: 10)[/]"
    assert (
        m.failed("msg", details=["a", "b"], exit_code=10)
        == f"{fail} msg{exit}\n   a\n   b"
    )


def test_skipped() -> None:
    assert m.skipped("msg") == "[cyan][SKIP][/] msg"


def test_timeout() -> None:
    assert m.timeout("msg") == "[yellow][TIME][/] msg"


def test_summary() -> None:
    assert m.summary("foo", 12, 11, timedelta(seconds=2.123)) == (
        f"[bright red]{'foo: Passed 11 of 12 tests (91.7%) in 2.12s': >{m.UI_WIDTH}}[/]"  # noqa E501
    )


def test_summary_no_justify() -> None:
    assert (
        m.summary("foo", 12, 11, timedelta(seconds=2.123), justify=False)
        == "foo: Passed 11 of 12 tests (91.7%) in 2.12s"
    )
