# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    import os
    from collections.abc import Callable
    from types import TracebackType

    from .log import BenchmarkLog


class BenchmarkLogFromFilename:
    """Wrap a BenchmarkLog in a context that also opens a file."""

    file_name: os.PathLike[str]
    constructor: Callable[[TextIO], BenchmarkLog]
    _stack: ExitStack

    def __init__(
        self,
        file_name: os.PathLike[str],
        constructor: Callable[[TextIO], BenchmarkLog],
    ) -> None:
        """Create a context manager that opens a file for a BenchmarkLog.

        This class exists so `benchmark_log()` can conditionally open output
        files, most users should use that function and not call this directly.

        Parameters
        ----------
        file_name: os.PathLike[str]
            The path for the file to open.
        constructor: Callable[[TextIO], BenchmarkLog]
            Thunk for creating a `BenchmarkLog` from the file handle that was
            opened from `file_name`.
        """
        self.file_name = file_name
        self.constructor = constructor

    def __enter__(self) -> BenchmarkLog:
        log: BenchmarkLog
        with ExitStack() as stack:
            file_handle = stack.enter_context(Path(self.file_name).open("w"))
            log = self.constructor(file_handle)
            log = stack.enter_context(log)
            self._stack = stack.pop_all()
        return log

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TracebackType | None,
    ) -> None:
        self._stack.__exit__(exc_type, exc_value, exc_traceback)
