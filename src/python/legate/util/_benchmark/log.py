# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import ExitStack
from typing import TYPE_CHECKING, Any, Self, TextIO

from ..info import _nested_dict_pretty_print, info as legate_info
from .info import benchmark_info

if TYPE_CHECKING:
    import numpy as np


class BenchmarkLog(ExitStack):
    """Base class to manage logging of benchmarking data."""

    name: str
    uid: np.uint64
    file: TextIO
    columns: list[str]

    def __init__(
        self, name: str, uid: np.uint64, columns: list[str], file: TextIO
    ) -> None:
        """Create a context manager for collecting benchmark data in a table.

        Most users should use `benchmark_log()` instead of calling this
        directly.

        Parameters
        ----------
        name: str
            The name of the benchmark.
        uid: np.uint64
            The unique identifier of the benchmark.
        columns: list[str]
            The names for the columns of data that will be collected.
        file: TextIO
            An output text stream to accept the data as it is collected.
        """
        super().__init__()
        self.name = name
        self.uid = uid
        self.columns = columns
        self.file = file

    def _log_metadata(self, _metadata: str) -> None:
        raise NotImplementedError

    def _log_columns(self, _columns: list[str]) -> None:
        raise NotImplementedError

    def _log_row(self, _row: list[str]) -> None:
        raise NotImplementedError

    def _get_row(self, row: dict[str, Any]) -> list[str]:
        return [
            (str(row[a]) if a in row else "(missing)") for a in self.columns
        ]

    def _generate_and_log_metadata(self) -> None:
        info: dict[str, Any] = {}
        info["Benchmark"] = benchmark_info(self.name, self.uid)
        info.update(legate_info())
        lines = "\n".join(_nested_dict_pretty_print(info))
        self._log_metadata(lines)

    def __enter__(self) -> Self:
        """Open a log file to accept rows recorded with `log()`."""
        super().__enter__()
        self._generate_and_log_metadata()
        self._log_columns(self.columns)
        return self

    def log(self, **kwargs: Any) -> None:
        """
        Add a row to a benchmark table created in `benchmark_log()`.

        Parameters
        ----------
        Use the columns specified in `benchmark_log()` as the arguments to
        `log()`, for example:

        ```
        from legate.util.benchmark import benchmark_log

        with benchmark_log("mybench", columns=["time", "size"]) as b:
            b.log(time=1.0, size=1000)
        ```

        If your columns are not valid identifiers, use unpacking:
        ```
        from legate.util.benchmark import benchmark_log

        time="Time (seconds)"
        size="Florps"

        with benchmark_log("mybench", columns=[time, size]) as b:
            b.log(**{time: 1.0, size: 1000})
        ```
        """
        row = self._get_row(kwargs)
        self._log_row(row)
