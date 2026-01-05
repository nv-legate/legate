# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import secrets
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

import numpy as np

from ..core import get_legate_runtime, types as ty
from ..core.task import OutputArray, task
from ..settings import settings as legate_settings
from ._benchmark.log_csv import BenchmarkLogCSV
from ._benchmark.log_from_filename import BenchmarkLogFromFilename
from ._benchmark.log_rich import BenchmarkLogRich
from ._benchmark.settings import settings

if TYPE_CHECKING:
    import os

    from ._benchmark.log import BenchmarkLog

__all__ = ["benchmark_log"]


def _num_nodes() -> int:
    runtime = get_legate_runtime()
    machine = runtime.get_machine()
    nodes = machine.get_node_range()
    return nodes[1] - nodes[0]


@task
def _pick_one_uid(uid: np.uint64, out: OutputArray) -> None:
    np.asarray(out)[:] = uid


def _benchmark_uid() -> np.uint64:
    """Create a random identifier that is the same on all ranks."""
    uid = np.uint64(int.from_bytes(secrets.token_bytes(8)))
    if _num_nodes() > 1:
        arr = np.array([0], dtype=np.uint64)
        store = get_legate_runtime().create_store_from_buffer(
            ty.uint64, arr.shape, arr, read_only=False
        )
        _pick_one_uid(uid, store)
        return np.asarray(store.get_physical_store())[0]
    return uid


def _benchmark_file(out: TextIO | None) -> TextIO | None:
    if out is not None:
        return out
    log_location = settings.out()
    if log_location == "stdout":
        return sys.stdout
    return None


def _benchmark_file_name(name: str, uid: np.uint64) -> os.PathLike[str]:
    log_location = settings.out()
    assert log_location != "stdout"
    name_start = name.replace(" ", "")
    local_name = f"{name_start}_{uid:016x}.{get_legate_runtime().node_id}.csv"
    return Path(log_location) / local_name


def _use_rich(out: TextIO) -> bool:
    if out.isatty() and settings.use_rich():
        # live updating from multiple ranks won't work, so require there to be
        # only one rank (or only one rank using stdout)
        return _num_nodes() == 1 or legate_settings.limit_stdout()
    return False


def benchmark_log(
    name: str, columns: list[str], out: TextIO | None = None
) -> BenchmarkLog | BenchmarkLogFromFilename:
    """
    Create a context manager for logging tables of data generated for
    benchmarking legate code.

    The context manager will write a table of benchmarking data to a specified
    output textstream, including with the table a header comment with
    reproducibility data about how the benchmark was run.

    Parameters
    ----------
    name: str
        The name for the benchmark.
    columns: list[str]
        A list of headers for the columns of data in the table.
    out: TextIO | None = None
        Optional io stream for benchmark data: e.g. `out=sys.stdout` to write
        benchmark data to the screen.  If `out` is not specified, the
        destination of benchmark data depends on the `LEGATE_BENCHMARK_OUT`
        environment variable.  By default, this variable is `stdout`, in which
        case benchmark data will be written to `sys.stdout` (see also
        `LEGATE_LIMIT_STDOUT`).  If instead this is a directory, e.g.
        `LEGATE_BENCHMARK_OUT=${PWD}`, then a unique basename will be generated
        for a set of output csv files (one per rank) in that directory.  For
        example, if `name` is `mybench`, then rank `P` will write its
        benchmark data to `mybench_[unique hex string].P.csv`.

    Returns
    -------
    BenchmarkLog | BenchmarkLogFromFilename
        A context manager whose one method is `log()`, which adds
        a row of benchmark data to the table.
    """
    uid = _benchmark_uid()
    file = _benchmark_file(out)
    if file is not None:
        if _use_rich(file):
            return BenchmarkLogRich(name, uid, columns, file)
        return BenchmarkLogCSV(name, uid, columns, file)

    file_name = _benchmark_file_name(name, uid)

    def thunk(file: TextIO) -> BenchmarkLog:
        return BenchmarkLogCSV(name, uid, columns, file)

    return BenchmarkLogFromFilename(file_name, thunk)
