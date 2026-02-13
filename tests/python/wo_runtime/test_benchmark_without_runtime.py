# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import io
import re

from legate.util.benchmark import benchmark_log
from legate.util.has_started import runtime_has_started


def test_metadata_without_runtime() -> None:
    outfile = io.StringIO()
    with benchmark_log(
        "basic", ["size", "time"], out=outfile, start_runtime=False
    ):
        pass
    assert not runtime_has_started()
    string = outfile.getvalue()
    infile = io.StringIO(string)
    header = filter(lambda r: r[0] == "#", infile)
    metadata = "".join(line.strip("#") for line in header)
    expected = [
        r"Legate runtime configuration *: *\(unavailable, legate runtime not started\)",  # noqa: E501
        r"Machine *: *\(unavailable, legate runtime not started\)",
    ]
    for entry in expected:
        assert re.search(entry, metadata) is not None
