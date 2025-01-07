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

"""Provide a basic logger that can scrub ANSI color codes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich import print as rich_print
from rich.console import Console

if TYPE_CHECKING:
    from rich.console import RenderableType


class Log:
    def __init__(self) -> None:
        self._console = Console(color_system=None, soft_wrap=True)
        self._record: list[str] = []

    def __call__(self, *lines: RenderableType | str) -> None:  # noqa: D102
        self.render(*lines)

    def render(self, *items: RenderableType | str) -> None:  # noqa: D102
        for item in items:
            rich_print(item, flush=True)
            with self._console.capture() as capture:
                self._console.print(item)
            self._record.extend(capture.get().strip().split("\n"))

    def clear(self) -> None:  # noqa: D102
        self._record = []

    def dump(self) -> str:  # noqa: D102
        return "\n".join(self._record)

    @property
    def lines(self) -> tuple[str, ...]:  # noqa: D102
        return tuple(self._record)


LOG = Log()
