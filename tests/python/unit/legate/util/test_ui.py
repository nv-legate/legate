# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import timedelta

from rich.text import Text

from legate.util import ui as m


def test_UI_WIDTH() -> None:
    assert m.UI_WIDTH == 80


def test_banner_simple() -> None:
    b = m.banner("some text", "and content")
    assert b.title == "some text"
    assert "and content" in str(b.renderable)


def test_error() -> None:
    assert m.error("some message") == Text.from_markup(
        "[red]ERROR:[/] some message"
    )


def test_section() -> None:
    s = m.section("some section")
    assert "some section" in str(s.renderable)


def test_warn() -> None:
    assert m.warn("some message") == Text.from_markup(
        "[magenta]WARNING:[/] some message"
    )


def test_shell() -> None:
    assert m.shell("cmd --foo") == Text("+cmd --foo", style="dim white")


def test_shell_with_char() -> None:
    assert m.shell("cmd --foo", char="") == Text(
        "cmd --foo", style="dim white"
    )


def test_passed() -> None:
    assert m.passed("msg") == Text.from_markup("[bold green][PASS][/] msg")


def test_passed_with_details() -> None:
    assert m.passed("msg", details=["a", "b"]) == Text.from_markup(
        "[bold green][PASS][/] msg\n   a\n   b\n"
    )


def test_failed() -> None:
    assert m.failed("msg") == Text.from_markup("[bold red][FAIL][/] msg")


def test_failed_with_exit_code() -> None:
    fail_str = "[bold red][FAIL][/]"
    exit_str = " [bold white](exit: 10)[/]"
    assert m.failed("msg", exit_code=10) == Text.from_markup(
        f"{fail_str} msg{exit_str}"
    )


def test_failed_with_details() -> None:
    assert m.failed("msg", details=["a", "b"]) == Text.from_markup(
        "[bold red][FAIL][/] msg\n   a\n   b\n"
    )


def test_failed_with_details_and_exit_code() -> None:
    fail_str = "[bold red][FAIL][/]"
    exit_str = " [bold white](exit: 10)[/]"
    assert m.failed(
        "msg", details=["a", "b"], exit_code=10
    ) == Text.from_markup(f"{fail_str} msg{exit_str}\n   a\n   b\n")


def test_skipped() -> None:
    assert m.skipped("msg") == Text.from_markup("[cyan][SKIP][/] msg")


def test_timeout() -> None:
    assert m.timeout("msg") == Text.from_markup("[yellow][TIME][/] msg")


def test_summary() -> None:
    assert m.summary(12, 11, timedelta(seconds=2.123)) == Text.from_markup(
        "[bold red]Passed 11 of 12 tests (91.7%) in 2.12s[/]"
    )
