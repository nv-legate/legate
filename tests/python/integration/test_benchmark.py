# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import io
import re
import csv
import sys
from typing import Any, TextIO

from legate.util.benchmark import benchmark_log


def read_commented_csv(f: TextIO) -> dict[str, list[str]]:
    d: dict[str, list[str]] = {}
    reader = csv.DictReader(filter(lambda row: row[0] != "#", f))
    for row in reader:
        for k, v in row.items():
            if k in d:
                d[k].append(v)
            else:
                d[k] = [v]
    return d


SIZES = [1, 10, 100, 1000]
TIMES = [0.5, 1.0, 2.0, 4.0]


def write_string_dict(**kwargs: list[Any]) -> dict[str, list[str]]:
    d: dict[str, list[str]] = {}
    for k, v in kwargs.items():
        d[k] = [str(i) for i in v]
    return d


def test_benchmark_basic() -> None:
    outfile = io.StringIO()
    d_direct = write_string_dict(size=SIZES, time=TIMES)
    with benchmark_log("basic", ["size", "time"], out=outfile) as b:
        for s, t in zip(SIZES, TIMES, strict=True):
            b.log(time=t, size=s)
    string = outfile.getvalue()
    infile = io.StringIO(string)
    d_from_file = read_commented_csv(infile)
    assert d_direct == d_from_file


def test_benchmark_rich() -> None:
    # this test only fails if rich emits an error, and you can
    # observe rich output with pytest options `-s -k test_benchmark_rich`,
    # but I'm not sure how to write a test to verify rich output
    with benchmark_log("basic", ["size", "time"], out=sys.stdout) as b:
        for s, t in zip(SIZES, TIMES, strict=True):
            b.log(time=t, size=s)


def test_benchmark_fields_with_spaces() -> None:
    outfile = io.StringIO()
    size = "size (florps)"
    time = "time (seconds)"
    d_direct = write_string_dict(**{size: SIZES, time: TIMES})  # type: ignore[arg-type]
    with benchmark_log("basic (with spaces)", [size, time], out=outfile) as b:
        for s, t in zip(SIZES, TIMES, strict=True):
            b.log(**{time: t, size: s})
    string = outfile.getvalue()
    infile = io.StringIO(string)
    d_from_file = read_commented_csv(infile)
    assert d_direct == d_from_file


def test_benchmark_empty() -> None:
    sizes_with_empty = ["1", "10", "(missing)", "(missing)"]
    times_with_empty = ["0.5", "(missing)", "2.0", "(missing)"]
    outfile = io.StringIO()
    d_direct = write_string_dict(size=sizes_with_empty, time=times_with_empty)
    with benchmark_log("basic", ["size", "time"], out=outfile) as b:
        b.log(size=1, time=0.5)
        b.log(size=10)
        b.log(time=2.0)
        b.log()
    string = outfile.getvalue()
    infile = io.StringIO(string)
    d_from_file = read_commented_csv(infile)
    assert d_direct == d_from_file


def test_metadata() -> None:
    outfile = io.StringIO()
    with benchmark_log("basic", ["size", "time"], out=outfile):
        pass
    string = outfile.getvalue()
    infile = io.StringIO(string)
    header = filter(lambda r: r[0] == "#", infile)
    metadata = "".join(line.strip("#") for line in header)
    expected = [
        "Benchmark *:",
        "  Name *:",
        "  ID *:",
        "  Time *:",
        "  Global rank *:",
        "  Local rank *:",
        "  Node *:",
        "Program *:",
        "Legate runtime configuration *:",
        "Machine *:",
        "  Preferred target *:",
        "  GPU *:",
        "  OMP *:",
        "  CPU *:",
        "System info *:",
        "  Python *:",
        "  Platform *:",
        "  GPU driver *:",
        "  GPU devices *:",
        "Package versions *:",
        "  legion *:",
        "  legate *:",
        "  cupynumeric *:",
        "  numpy *:",
        "  scipy *:",
        "  numba *:",
        "Package details *:",
        "  cuda-version *:",
        "  legate *:",
        "  cupynumeric *:",
        "Legate build configuration *:",
        "  build_type *:",
        "  use_openmp *:",
        "  use_cuda *:",
        "  networks *:",
        "  conduit *:",
        "  configure_options *:",
    ]
    for entry in expected:
        assert re.search(entry, metadata) is not None


def test_custom_metadata() -> None:
    outfile = io.StringIO()
    input_metadata = {
        "Custom Field": {"Custom Subfield 1": 0, "Custom Subfield 2": 1}
    }
    with benchmark_log(
        "basic", ["size", "time"], out=outfile, metadata=input_metadata
    ):
        pass
    string = outfile.getvalue()
    infile = io.StringIO(string)
    header = filter(lambda r: r[0] == "#", infile)
    metadata = "".join(line.strip("#") for line in header)
    expected = [
        "Benchmark *:",
        "  Name *:",
        "  ID *:",
        "  Time *:",
        "  Global rank *:",
        "  Local rank *:",
        "  Node *:",
        "Custom Field *:",
        "  Custom Subfield 1 *:",
        "  Custom Subfield 2 *:",
    ]
    for entry in expected:
        assert re.search(entry, metadata) is not None
