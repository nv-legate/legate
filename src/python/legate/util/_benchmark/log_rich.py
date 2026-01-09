# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Self, TextIO

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from .log import BenchmarkLog

if TYPE_CHECKING:
    import numpy as np


class BenchmarkLogRich(BenchmarkLog):
    """A class for pretty-printing benchmark data to the screen."""

    def __init__(
        self, name: str, uid: np.uint64, columns: list[str], file: TextIO
    ) -> None:
        """
        Create a context manager to pretty-print benchmark data to the screen.

        See `BenchmarkLog()` for details on the arguments.

        Most users should just call `benchmark_log()` instead of calling this
        directly.
        """
        assert file.isatty()
        super().__init__(name, uid, columns, file)
        self.console = Console(file=file)
        self.table = Table(title=f"Benchmark [bold]{name}[/bold] Data")

    def __enter__(self) -> Self:
        super().__enter__()
        self.enter_context(
            Live(self.table, console=self.console, refresh_per_second=1)
        )
        return self

    def _log_metadata(self, metadata: str) -> None:
        rendered = self.console.render_str(metadata)
        title = f"Benchmark [bold]{self.name}[/] Configuration Details"
        self.console.print(Panel(rendered, title=title))

    def _log_row(self, row: list[str]) -> None:
        self.table.add_row(*tuple(self.console.render_str(r) for r in row))

    def _log_columns(self, columns: list[str]) -> None:
        for column in columns:
            self.table.add_column(column)
