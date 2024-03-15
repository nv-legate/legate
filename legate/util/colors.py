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

"""Helper functions for adding colors to simple text UI output.

The color functions in this module require ``colorama`` to be installed in
order to generate color output. If ``colorama`` is not available, plain
text output (i.e. without ANSI color codes) will be generated.

"""
from __future__ import annotations

import re
import sys

__all__ = (
    "bright",
    "cyan",
    "dim",
    "green",
    "magenta",
    "red",
    "scrub",
    "white",
    "yellow",
    "HAVE_COLOR",
)


# Color terminal output needs to be explicitly opt-in. Applications that want
# to enable it should set this global flag to True, e.g based on a command line
# argument or other user-supplied configuration
ENABLED = False


def _text(text: str) -> str:
    return text


try:
    # Need to add unused-ignore here since mypy complains:
    #
    # Unused "type: ignore" comment, use narrower [import-untyped] instead of
    # [import] code
    #
    # But colorama may or may not be installed, and so the suggested narrower
    # error code ends up being wrong half the time.
    import colorama  # type: ignore[import, unused-ignore]

    def bright(text: str) -> str:
        if not ENABLED:
            return text
        return f"{colorama.Style.BRIGHT}{text}{colorama.Style.RESET_ALL}"

    def dim(text: str) -> str:
        if not ENABLED:
            return text
        return f"{colorama.Style.DIM}{text}{colorama.Style.RESET_ALL}"

    def white(text: str) -> str:
        if not ENABLED:
            return text
        return f"{colorama.Fore.WHITE}{text}{colorama.Style.RESET_ALL}"

    def cyan(text: str) -> str:
        if not ENABLED:
            return text
        return f"{colorama.Fore.CYAN}{text}{colorama.Style.RESET_ALL}"

    def red(text: str) -> str:
        if not ENABLED:
            return text
        return f"{colorama.Fore.RED}{text}{colorama.Style.RESET_ALL}"

    def magenta(text: str) -> str:
        if not ENABLED:
            return text
        return f"{colorama.Fore.MAGENTA}{text}{colorama.Style.RESET_ALL}"

    def green(text: str) -> str:
        if not ENABLED:
            return text
        return f"{colorama.Fore.GREEN}{text}{colorama.Style.RESET_ALL}"

    def yellow(text: str) -> str:
        if not ENABLED:
            return text
        return f"{colorama.Fore.YELLOW}{text}{colorama.Style.RESET_ALL}"

    if sys.platform == "win32":
        colorama.init()

    HAVE_COLOR = True
except ImportError:
    bright = dim = white = cyan = red = magenta = green = yellow = _text
    HAVE_COLOR = False

# ref: https://stackoverflow.com/a/14693789
_ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def scrub(text: str) -> str:
    """Remove ANSI color codes from a text string.

    Parameters
    ----------
    text : str
        The text to scrub

    Returns
    -------
        str

    """
    return _ANSI_ESCAPE.sub("", text)
