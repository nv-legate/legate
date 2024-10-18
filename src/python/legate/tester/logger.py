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

from rich import print as rich_print
from rich.console import Console


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
            self._record.append(line)
            rich_print(line, flush=True)
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
        lines = self._record[start:end]
        with console.capture() as capture:
            console.print(*lines, sep="\n", end="")
        return capture.get()

    @property
    def lines(self) -> tuple[str, ...]:
        return tuple(self._record)


LOG = Log()
