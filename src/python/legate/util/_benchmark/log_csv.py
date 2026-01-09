# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
from textwrap import indent
from typing import TYPE_CHECKING, TextIO

from .log import BenchmarkLog

if TYPE_CHECKING:
    import numpy as np


class BenchmarkLogCSV(BenchmarkLog):
    def __init__(
        self, name: str, uid: np.uint64, columns: list[str], file: TextIO
    ) -> None:
        """
        Create a context for logging benchmark information to a .csv file.

        See `BenchmarkLog()` for details on the arguments.

        Most users should just call `benchmark_log()` instead of calling this
        directly.
        """
        super().__init__(name, uid, columns, file)
        self.csv = csv.writer(file, dialect="unix", quoting=csv.QUOTE_MINIMAL)

    def _log_metadata(self, metadata: str) -> None:
        COLS = 80
        commented_lines = indent(metadata, "# ")
        self.file.write("#" * COLS + "\n")
        self.file.write(commented_lines + "\n")
        self.file.write("#" * COLS + "\n")

    def _log_row(self, row: list[str]) -> None:
        self.csv.writerow(row)

    def _log_columns(self, columns: list[str]) -> None:
        self._log_row(columns)
