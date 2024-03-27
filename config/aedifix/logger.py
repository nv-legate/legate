# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.B
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

import logging
import os
import shutil
import sys
import textwrap
from pathlib import Path
from typing import Any, ClassVar, TypeVar

from .util.constants import Constants

_T = TypeVar("_T")


class Logger:
    __slots__ = ("_screen_logger", "_file_logger")
    __unique_id: ClassVar = 0

    def __init__(self, path: Path) -> None:
        r"""Construct a Logger.

        Parameters
        ----------
        path : Path
            The path at which to create the on-disk log.
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

        path = path.resolve()
        self._screen_logger = self._create_logger(
            make_name("screen_configure"),
            logging.StreamHandler,
            stream=sys.stdout,
        )
        self._file_logger = self._create_logger(
            make_name("file_configure"),
            logging.FileHandler,
            path,
            mode="w",
            delay=True,
        )

    def __del__(self) -> None:
        for attr in ("_screen_logger", "_file_logger"):
            logger = getattr(self, attr, None)
            if logger is None:
                continue
            for handler in logger.handlers:
                try:
                    handler.flush()
                except ValueError:
                    pass

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
        setattr(func, "__config_log_ignore___", True)
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

    @staticmethod
    def build_multiline_message(
        sup_title: str,
        text: str,
        divider_char: str | None = None,
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
        divider_shar : str, optional
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
            length = Constants.banner_length

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
        if divider_char is not None:
            # add the divider if we are making a message like
            #
            # =====================
            #   BIG SCARY TITLE
            # --------------------- <- divider_char is '-'
            #   foo bar
            divider_char = str(divider_char)
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

    @classmethod
    def build_multiline_error_message(
        cls,
        sup_title: str,
        text: str,
        divider_char: str = "-",
        length: int | None = None,
        **kwargs: Any,
    ) -> str:
        r"""Build a multiline error message.

        Parameters
        ----------
        sup_tile : str
            The super title for the message, e.g. 'WARNING'.
        text : str
            The body of the message.
        divider_shar : str, '-'
            The char to form the divider between `sup_title` and `text`, if
            any. Must be a single character. Defaults to '-'.
        length : int, optional
            The maximum width of the message. Defaults to the width of the
            screen.
        **kwargs : Any
            Additional keyword arguments to `Logger.build_multiline_message()`.

        Returns
        -------
        message : str
            The formatted error message.
        """
        if length is None:
            length = Constants.banner_length

        if not text.endswith("\n"):
            text += "\n"

        banner_line = "*" * length
        return "\n".join(
            [
                banner_line,
                cls.build_multiline_message(
                    sup_title,
                    text,
                    divider_char=divider_char,
                    length=length,
                    **kwargs,
                ),
                banner_line,
                "",  # to add an additional newline at the end
            ]
        )

    def flush(self) -> None:
        r"""Flush any pending log writes to disk or screen."""
        for logger in (self._file_logger, self._screen_logger):
            for handler in logger.handlers:
                try:
                    handler.flush()
                except AttributeError:
                    pass

    def log_screen(
        self, mess: str, end: str = "\n", flush: bool = False
    ) -> None:
        r"""Log a message to the screen.

        Parameters
        ----------
        mess : str
            The message to print to screen.
        end : str, '\n'
            The line ending to append to mess. Defaults to newline.
        flush : bool, False
            True if the log handler should flush before returning, False
            otherwise.
        """
        if end == "\r":
            flush = True
        was_scrolling = False
        slogger = self._screen_logger
        handlers = slogger.handlers
        for handler in handlers:
            assert isinstance(handler, logging.StreamHandler)
            if handler.terminator == "\r":
                was_scrolling = True
            handler.terminator = end

        # Kind of a dirty hack to re-stablish newlines after "scrolling"
        # behavior.
        if end == "\n" and was_scrolling:
            mess = "\n" + mess
        if end not in {"\n", "\r"}:
            mess += end
        slogger.log(slogger.level, mess)
        if flush:
            for handler in handlers:
                try:
                    handler.flush()
                except AttributeError:
                    pass

    def log_screen_clear_line(self) -> None:
        r"""Clear the current line for the screen logger."""
        self.log_screen("\r\033[K", end="\r")

    def log_file(self, message: str) -> None:
        r"""Log a message to the log file.

        Parameters
        ----------
        message : str
            The message to log to file.
        """
        self._file_logger.log(self._screen_logger.level, message)

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
                f"Destination log path ({dest}) same as source, "
                "not copying!"
            )
            return dest
        self.log_file(f"Copying file log from {src} to {dest}")
        self.flush()
        return Path(shutil.copy2(src, dest))
