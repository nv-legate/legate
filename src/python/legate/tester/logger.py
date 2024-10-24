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

"""Provide a basic logger that can scrub ANSI color codes.

"""
from __future__ import annotations

import re

from rich import print as rich_print
from rich.console import Console
from rich.text import Text

# ref: https://stackoverflow.com/a/14693789
_ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def _has_ansi(line: str) -> bool:
    return _ANSI_ESCAPE.search(line) is not None


def _scrub_ansi(line: str) -> str:
    return _ANSI_ESCAPE.sub("", line)


class Log:
    def __init__(self) -> None:
        self._record: list[str] = []

    def __call__(self, *lines: str) -> tuple[int, int]:
        return self.record(*lines)

    def record(self, *lines: str) -> tuple[int, int]:
        if len(lines) == 1 and "\n" in lines[0]:
            lines = tuple(lines[0].split("\n"))
        start = len(self._record)
        for line in lines:
            # May need to call Text.from_ansi in case output from a subprocess
            # already contains raw ANSI color codes, e.g. gtest results.
            # Unfortunately, we have to apply a heuristic check for ANSI codes
            # on a per-line basis, since calling from_ansi unconditionally
            # would also cause rich markup stop functioning on other lines.
            if _has_ansi(line):
                text = Text.from_ansi(line, no_wrap=True)
                rich_print(text, flush=True)
            else:
                rich_print(line, flush=True)
            self._record.append(line)
        return (start, len(self._record))

    def clear(self) -> None:
        self._record = []

    def dump(
        self,
        *,
        start: int = 0,
        end: int | None = None,
    ) -> str:
        console = Console(color_system=None, soft_wrap=True)
        lines = (_scrub_ansi(line) for line in self._record[start:end])
        with console.capture() as capture:
            console.print(*lines, sep="\n", end="")
        return capture.get()

    @property
    def lines(self) -> tuple[str, ...]:
        return tuple(self._record)


LOG = Log()
