# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.B
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import sys
import shutil
import logging
import textwrap
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

# This must be the ONLY place that rich is imported to ensure that this error
# message is seen when running configure on a system where it is not yet
# installed.
try:
    import rich  # noqa: F401
except ModuleNotFoundError as mnfe:
    msg = "Please run 'python3 -m pip install rich' to continue"
    raise RuntimeError(msg) from mnfe

import contextlib

from rich.align import Align, AlignMethod
from rich.console import Console, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from typing_extensions import Self

if TYPE_CHECKING:
    from collections.abc import Sequence

_T = TypeVar("_T")


class Logger:
    __slots__ = (
        "_console",
        "_file_logger",
        "_live",
        "_live_raii",
        "_row_data",
        "_table",
    )
    __unique_id: ClassVar = 0

    def __init__(self, path: Path, max_live_lines: int = 40) -> None:
        r"""Construct a Logger.

        Parameters
        ----------
        path : Path
            The path at which to create the on-disk log.
        max_live_lines : 40
            The maximum number of live output lines to keep.
        """

        def make_name(base_name: str) -> str:
            env_var = "__AEDIFIX_TESTING_DO_NOT_USE_OR_YOU_WILL_BE_FIRED__"
            if os.environ.get(env_var, "") == "1":
                from random import random

                assert "pytest" in sys.modules, (
                    "Attempting to randomize the logger names outside of "
                    f"testing! The variable is called '{env_var}' for a "
                    "reason!"
                )
                base_name += f"_{int(random() * 100_000)}_{Logger.__unique_id}"
                Logger.__unique_id += 1
            return base_name

        self._file_logger = self._create_logger(
            make_name("file_configure"),
            logging.FileHandler,
            path.resolve(),
            mode="w",
            delay=True,
        )

        self._row_data: deque[tuple[RenderableType, bool]] = deque(
            maxlen=max_live_lines
        )
        self._console = Console()
        self._table = self._make_table(self._row_data)
        self._live = Live(
            self._table, console=self.console, auto_refresh=False
        )
        self._live_raii: Live | None = None

        orig_hook = sys.breakpointhook

        def bphook(*args: Any, **kwargs: Any) -> Any:
            self._live.stop()
            return orig_hook(*args, **kwargs)

        sys.breakpointhook = bphook

    @staticmethod
    def _make_table(row_data: deque[tuple[RenderableType, bool]]) -> Table:
        table = Table.grid(expand=True)
        table.highlight = True
        for data, _ in row_data:
            table.add_row(data)
        return table

    def __enter__(self) -> Self:
        self._live_raii = self._live.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        self.flush()
        self._live.__exit__(*args)  # type: ignore[arg-type]
        self._live_raii = None

    @property
    def console(self) -> Console:
        r"""Get the current active Console.

        Returns
        -------
        Console
            The current active console.
        """
        return self._console

    @property
    def file_path(self) -> Path:
        r"""Retrieve the path to the file handler log file.

        Returns
        -------
        file_path : Path
            The path to the file handler log file, e.g.
            '/path/to/configure.log'.
        """
        handlers = self._file_logger.handlers
        assert len(handlers) == 1, f"Multiple file handlers: {handlers}"
        assert isinstance(handlers[0], logging.FileHandler)
        return Path(handlers[0].baseFilename)

    @staticmethod
    def log_passthrough(func: _T) -> _T:
        r"""A decorator to signify that `func` should never appear in the log
        context.

        Parameters
        ----------
        func : T
            The function.

        Returns
        -------
        func : T
            `func` unchanged.

        Notes
        -----
        The logger usually prints the name of the calling function as the
        prefix of the logged message. It does this by walking up the call-stack
        until it finds an appropriate name to print. This decorator marks the
        decorated functions as ignored in this stack walk, i.e. the logger
        skips that function and keeps walking up the stack.

        This decorator is useful for defining "pass-through" function (hence
        the name), whose only job is to accept some arguments and forward them
        on to the next one. Such functions are effectively syntactic sugar, and
        hence should not count as the origin of the logged message. Consider
        for example:

        def foo():
            manager.log("hello")

        @Logger.log_passthrough
        def bar():
            manager.log("there")

        def baz():
            bar()

        >>> foo()
        <locals>.foo: 'hello'
        >>> baz()
        <locals>.baz: 'there'

        Note how bar() (the true originator of the logging call) is ignored,
        and baz is printed instead.
        """
        func.__config_log_ignore___ = True  # type: ignore[attr-defined]
        return func

    @staticmethod
    def _create_logger(
        name: str, HandlerType: type, *args: Any, **kwargs: Any
    ) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        handler = HandlerType(*args, **kwargs)
        handler.setLevel(logger.level)
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def build_multiline_message(
        self,
        sup_title: str,
        text: str,
        divider_char: str = "-",
        length: int | None = None,
        prefix: str = " " * 2,
        **kwargs: Any,
    ) -> str:
        r"""Construct a properly formatted multiline message.

        Parameters
        ----------
        sup_tile : str
            The super title for the message, e.g. 'WARNING'.
        text : str
            The body of the message.
        divider_char : str, "-"
            The char to form the divider between `sup_title` and `text`, if
            any. Must be a single character.
        length : int, optional
            The maximum width of the message.
        prefix : str, '  '
            The prefix to add to the beginning of each wrapped line of the
            resultant message.
        **kwargs : Any
            Additional keyword arguments to `textwrap.wrap()`.

        Returns
        -------
        message : str
            The formatted multiline message.
        """
        if length is None:
            length = self.console.width - 1

        kwargs.setdefault("break_on_hyphens", False)
        kwargs.setdefault("break_long_words", False)
        kwargs.setdefault("width", length - 2)
        kwargs.setdefault("initial_indent", prefix)
        kwargs.setdefault("subsequent_indent", prefix)
        kwargs.setdefault("drop_whitespace", False)

        def center_line(line: str) -> str:
            return line.center(length).rstrip()

        def add_dummy_space(line: str) -> str:
            # The extra " " returned is so that newlines ("\n\n") in text are
            # respected. splitlines() returns an empty string ("") for them:
            #
            # >>> x = "foo\n\nbar".splitlines()
            # ["foo", "", "bar"]
            #
            # (which is what we want), but textwrap.wrap() will simply discard
            # these empty strings:
            #
            # >>> textwrap.wrap(x)
            # []
            #
            # when instead we want it to return [""]. If we return " " (and set
            # drop_whitespace=False), then textwrap.wrap() returns [" "] as
            # expected.
            #
            # The extra whitespace is then taken care of in line.rstrip() later
            # in the list comprehension (which leaves the prefix intact).
            return line if line else " "

        wrapped = [
            line.rstrip()
            for para in text.splitlines()
            for line in textwrap.wrap(
                add_dummy_space(textwrap.dedent(para)), **kwargs
            )
        ]
        if len(wrapped) == 1:
            # center-justify single lines, and remove the bogus prefix
            wrapped[0] = center_line(wrapped[0].lstrip())
        if divider_char and sup_title:
            # add the divider if we are making a message like
            #
            # =====================
            #   BIG SCARY TITLE
            # --------------------- <- divider_char is '-'
            #   foo bar
            assert len(divider_char) == 1
            wrapped.insert(0, divider_char * length)
        if sup_title:
            # add the super title if we are making a message like
            #
            # =====================
            #   BIG SCARY TITLE     <- sup_title is 'BIG SCARY TITLE'
            # ---------------------
            #   foo bar
            # add the banner
            wrapped.insert(0, center_line(str(sup_title)))
        return "\n".join(wrapped)

    def flush(self) -> None:
        r"""Flush any pending log writes to disk or screen."""
        for handler in self._file_logger.handlers:
            with contextlib.suppress(AttributeError):
                handler.flush()
        self._live.refresh()

    def _append_live_message(
        self, mess: RenderableType, *, keep: bool
    ) -> None:
        r"""Append a new message to the live data stream.

        Parameters
        ----------
        mess : RenderableType
            The message to append.
        keep : bool
            True if the message should persist, False otherwise. If the
            message persists, then appending new data (which might otherwise
            bump the message off the queue) will instead bump the next
            available message off.

        """
        row_data = self._row_data
        assert row_data.maxlen is not None  # mypy
        if len(row_data) >= (row_data.maxlen - 1):
            for idx, (_, data_keep) in enumerate(row_data):
                if not data_keep:
                    del row_data[idx]
                    break
            else:
                msg = (
                    "Could not prune row data, every entry was marked as "
                    "persistent"
                )
                raise ValueError(msg)
        row_data.append((mess, keep))

    def log_screen(
        self,
        mess: (
            RenderableType | list[RenderableType] | tuple[RenderableType, ...]
        ),
        *,
        keep: bool = False,
    ) -> None:
        r"""Log a message to the screen.

        Parameters
        ----------
        mess : RenderableType |
               list[RenderableType] |
               tuple[RenderableType, ...]
            The message(s) to print to screen.
        keep : bool, False
            Whether to keep the message permanently in live output.
        """
        if not self._live.is_started:
            with self:
                self.log_screen(mess, keep=keep)
            return

        def do_log(message: RenderableType, *, keep: bool) -> None:
            self._append_live_message(message, keep=keep)
            self._table = self._make_table(self._row_data)
            self._live.update(self._table, refresh=True)

        match mess:
            case list() | tuple():
                for m in mess:
                    do_log(m, keep=keep)
            case _:
                do_log(mess, keep=keep)

    def log_file(self, message: str | Sequence[str]) -> None:
        r"""Log a message to the log file.

        Parameters
        ----------
        message : str | Sequence[str]
            The message, or sequence of lines to log to file.
        """
        if not isinstance(message, str):
            message = "\n".join(message)
        self._file_logger.log(self._file_logger.level, message)

    def _log_boxed_file(self, message: str, title: str) -> None:
        file_msg = self.build_multiline_message(title, message)
        self.log_divider(tee=False)
        self.log_file(file_msg)
        self.log_divider(tee=False)

    def _log_boxed_screen(
        self, message: str, title: str, style: str, align: AlignMethod
    ) -> None:
        def fixup_title(title: str, style: str) -> str:
            if not title:
                return title

            if style:
                if not style.startswith("["):
                    style = "[" + style
                if not style.endswith("]"):
                    style += "]"
            return f"[bold]{style}{title}[/]"

        title = fixup_title(title, style)
        rich_txt = self.console.render_str(message)
        screen_message = Panel(
            Align(rich_txt, align=align), style=style, title=title
        )
        self.log_screen(screen_message, keep=True)

    def log_boxed(
        self,
        message: str,
        *,
        title: str = "",
        title_style: str = "",
        align: AlignMethod = "center",
    ) -> None:
        r"""Log a message surrounded by a box.

        Parameters
        ----------
        message : str
            The message to log.
        title : str, ''
            An optional title for the box.
        title_style : str, ''
            Optional additional styling for the title.
        align : AlignMethod, 'center'
            How to align the text.
        """
        self._log_boxed_file(message, title)
        self._log_boxed_screen(message, title, title_style, align)

    def log_warning(self, message: str, *, title: str = "WARNING") -> None:
        r"""Log a warning to the log.

        Parameters
        ----------
        message : str
            The message to print.
        title : str, 'WARNING'
            The title to use for the box.
        """
        self.log_boxed(
            message,
            title=f"***** {title.strip()} *****",
            title_style="bold yellow",
        )

    def log_error(self, message: str, *, title: str = "WARNING") -> None:
        r"""Log a warning to the log.

        Parameters
        ----------
        message : str
            The message to print.
        title : str, 'WARNING'
            The title to use for the box.
        """
        self.log_boxed(
            message,
            title=f"***** {title.strip()} *****",
            title_style="bold red",
        )

    def log_divider(self, *, tee: bool = False, keep: bool = True) -> None:
        r"""Append a dividing line to the logs.

        Parameters
        ----------
        tee : bool, False
           If True, output is printed to screen in addition to being appended
           to the on-disk log file. If False, output is only written to disk.
        keep : bool, True
           If ``tee`` is True, whether to persist the message in terminal
           output.
        """
        self.log_file("=" * (self.console.width - 1))
        if tee:
            self.log_screen(Rule(), keep=keep)

    def copy_log(self, dest: Path) -> Path:
        r"""Copy the file log to another location.

        Parameters
        ----------
        dest : Path
            The destination to copy the log to.

        Returns
        -------
        dest : Path
            The destination path.
        """
        dest = dest.resolve()
        src = self.file_path
        if src == dest:
            self.log_file(
                f"Destination log path ({dest}) same as source, not copying!"
            )
            return dest
        self.log_file(f"Copying file log from {src} to {dest}")
        self.flush()
        return Path(shutil.copy2(src, dest))
