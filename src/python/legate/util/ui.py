# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helper functions for simple text UI output.

The color functions in this module require ``rich`` to be installed in
order to generate color output. If ``rich`` is not available, plain
text output (i.e. without ANSI color codes) will be generated.

"""

from __future__ import annotations

import shlex
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from datetime import timedelta

    from rich.console import RenderableType
    from rich.text import TextType

Details: TypeAlias = Iterable[str]
KeyVals: TypeAlias = dict[str, Any]
Justify: TypeAlias = Literal["default", "left", "center", "right", "full"]

ERROR = Text.from_markup("[red]ERROR:[/]")

__all__ = (
    "UI_WIDTH",
    "banner",
    "env",
    "error",
    "failed",
    "passed",
    "skipped",
    "table",
    "timeout",
    "warn",
)


#: Width for terminal output headers and footers.
UI_WIDTH = 80


def _format_details(
    details: Details, *, pre: Text | None = None, indent: int = 3
) -> Text:
    pad = " " * indent
    text = Text("\n")
    for line in details:
        if pre:
            text.append(pre)
        text += text.assemble(pad, Text.from_ansi(line, no_wrap=True), "\n")
    return text


def banner(
    title: TextType, content: RenderableType, *, width: int = UI_WIDTH
) -> Panel:
    """Generate a titled banner, with details included.

    Parameters
    ----------
    title : TextSType
        Renderable to use for the title.

    content : RenderableType,
        Content to display in area below the heading.

    width : int, optional
        How wide to draw the banner.

    """
    return Panel(
        content,
        title=title,
        title_align="left",
        padding=1,
        width=width,
        border_style="dim",
        box=box.DOUBLE_EDGE,
    )


def section(content: RenderableType, *, width: int = UI_WIDTH) -> Panel:
    """Generate a section divider.

    Parameters
    ----------
    content : RenderableType
        Renderable to use for the section divider

    width : int, optional
        How wide to draw the banner.

    """
    return Panel(content, width=width, border_style="dim")


def error(message: str, *, details: Details | None = None) -> Text:
    """Format a message as an error, including optional details.

    Parameters
    ----------
    message : str
        The message text to format after "ERROR".


    details : Details, optional
        A sequence of text lines to display below the message.

        The text may contain ANSI (e.g. if it comes from a subprocess), but
        is assumed not contain Rich markup tags.

    Returns
    -------
        Text

    """
    error = Text.from_markup("[red]ERROR:[/]")
    msg = Text.from_ansi(message)
    text = Text.assemble(error, " ", msg)
    if details:
        text += Text.assemble(error, "\n")
        text += _format_details(details, pre=error)
        text += Text.assemble(error, "\n")
    return text


def warn(message: str) -> Text:
    """Format a message as a warning.

    Parameters
    ----------
    message : str
        The message text to format after "WARNING".

    Returns
    -------
        Text

    """
    warning = Text.from_markup("[magenta]WARNING:[/]")
    msg = Text.from_markup(message)
    return Text.assemble(warning, " ", msg)


def skipped(message: str) -> Text:
    """Report a skipped test with a cyan [SKIP].

    Parameters
    ----------
    message : str
        Message text to format after "[SKIP]".

    Returns
    -------
        Text

    """
    skipped = Text.from_markup("[cyan][SKIP][/]")
    msg = Text.from_markup(message)
    return Text.assemble(skipped, " ", msg)


def timeout(message: str, *, details: Details | None = None) -> Text:
    """Report a timed-out test with a yellow [TIME].

    Parameters
    ----------
    message : str
        Message text to format after "[TIME]".

    details : Details, optional
        A sequence of text lines to display below the message.

        The text may contain ANSI (e.g. if it comes from a subprocess), but
        is assumed not contain Rich markup tags.

    Returns
    -------
        Text

    """
    timeout = Text.from_markup("[yellow][TIME][/]")
    msg = Text.from_markup(message)
    text = Text.assemble(timeout, " ", msg)
    if details:
        text += _format_details(details)
    return text


def failed(
    message: str,
    *,
    details: Details | None = None,
    exit_code: int | None = None,
) -> Text:
    """Report a failed test result with a bold red [FAIL].

    Parameters
    ----------
    message : str
        Message text to format after "[FAIL]".

    details : Iterable[str], optional
        A sequenece of text lines to display below the message.

        The text may contain ANSI (e.g. if it comes from a subprocess), but
        is assumed not contain Rich markup tags.

    Returns
    -------
        Text

    """
    failed = Text.from_markup("[bold red][FAIL][/]")
    msg = Text.from_markup(message)
    text = Text.assemble(failed, " ", msg)
    if exit_code is not None:
        text += Text.from_markup(f" [bold white](exit: {exit_code})[/]")
    if details:
        text += _format_details(details)
    return text


def passed(message: str, *, details: Details | None = None) -> Text:
    """Report a passed test result with a bold green [PASS].

    Parameters
    ----------
    message : str
        Message text to format after "[PASS]".

    details : Iterable[str], optional
        A sequenece of text lines to display below the message.

        The text may contain ANSI (e.g. if it comes from a subprocess), but
        is assumed not contain Rich markup tags.

    Returns
    -------
        Text

    """
    passed = Text.from_markup("[bold green][PASS][/]")
    msg = Text.from_markup(message)
    text = Text.assemble(passed, " ", msg)
    if details:
        text += _format_details(details)
    return text


def table(
    items: KeyVals, *, quote: bool = True, justify: Justify = "right"
) -> Table:
    """Format a dictionary as a basic table.

    By default, values are passed to shlex.quote before formatting.

    Parameters
    ----------
    items : dict[str, Any]
        The dictionary of items to format

    Returns
    -------
        Table

    """
    table = Table(box=None, show_header=False)

    table.add_column("Keys", justify=justify, style="dim green", no_wrap=True)
    table.add_column("Values", style="yellow", no_wrap=True)

    for key in items:
        val = shlex.quote(str(items[key])) if quote else str(items[key])
        table.add_row(key, val)

    return table


def env(items: KeyVals, *, keys: Iterable[str] | None = None) -> Text:
    """Format a dictionary as a table of environment variables.

    Values are passed to shlex.quote before formatting.

    Parameters
    ----------
    items : dict[str, Any]
        The dictionary of items to format

    keys : Iterable[str] or None, optional
        If not None, only the specified subset of keys is included in the
        output (default: None)

    Returns
    -------
        Text

    """
    keys = items.keys() if keys is None else keys

    text = Text()

    # don't use Text.from_markup here since keys or values
    # might contain content that could confuse rich
    for key in keys:
        k = Text(f" {key}", style="dim green")
        v = Text(shlex.quote(str(items[key])), style="yellow")
        text += Text.assemble(k, "=", v, "\n")

    return text


def shell(cmd: str, *, char: str = "+") -> Text:
    """Report a shell command in a dim white color.

    Parameters
    ----------
    cmd : str
        The shell command string to display

    char : str, optional
        A character to prefix the ``cmd`` with. (default: "+")

    Returns
    -------
        Text

    """
    # don't use Text.from_markup here since cmd
    # might contain content that could confuse rich
    return Text(f"{char}{cmd}", style="dim white")


def summary(total: int, passed: int, time: timedelta) -> Text:
    """Generate a test result summary line.

    The output is bold green if all tests passed, otherwise bold red.

    Parameters
    ----------
    total : int
        The total number of tests to report.

    passed : int
        The number of passed tests to report.

    time : timedelta
        The time taken to run the tests

    Returns
    -------
        Text

    """
    if total == 0:
        return Text.from_markup("[bold red]No tests run, please check[/]")

    color = "green" if passed == total else "red"

    summary = (
        f"Passed {passed} of {total} tests "
        f"({passed / total * 100:0.1f}%) "
        f"in {time.total_seconds():0.2f}s"
    )

    return Text.from_markup(f"[bold {color}]{summary}[/]")
