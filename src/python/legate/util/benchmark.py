# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import secrets
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO, cast

import numpy as np

from ..settings import settings as legate_settings
from ._benchmark.log import BenchmarkLog
from ._benchmark.log_csv import BenchmarkLogCSV
from ._benchmark.log_from_filename import BenchmarkLogFromFilename
from ._benchmark.log_rich import BenchmarkLogRich
from ._benchmark.settings import settings
from .has_started import runtime_has_started
from .info import info as legate_info

if TYPE_CHECKING:
    import os


__all__ = ["BenchmarkLog", "BenchmarkLogFromFilename", "benchmark_log"]


def _node_info() -> tuple[int, int]:
    # local import to avoid starting the runtime unless requested
    from ..core import get_legate_runtime  # noqa: PLC0415

    runtime = get_legate_runtime()
    machine = runtime.get_machine()
    nodes = machine.get_node_range()
    return (nodes[1] - nodes[0], runtime.node_id)


def _benchmark_uid() -> np.uint64:
    """Create a random identifier."""
    return np.uint64(int.from_bytes(secrets.token_bytes(8)))


def _benchmark_file(out: TextIO | None) -> TextIO | None:
    if out is not None:
        return out
    log_location = settings.out()
    if log_location == "stdout":
        return sys.stdout
    return None


def _benchmark_file_name(
    name: str, uid: np.uint64, node_id: int
) -> os.PathLike[str]:
    log_location = settings.out()
    assert log_location != "stdout"
    name_start = name.replace(" ", "")
    local_name = f"{name_start}_{uid:016x}.{node_id}.csv"
    return Path(log_location) / local_name


def _use_rich(out: TextIO, num_nodes: int) -> bool:
    if out.isatty() and settings.use_rich():
        # live updating from multiple ranks won't work, so require there to be
        # only one rank (or only one rank using stdout)
        return num_nodes == 1 or legate_settings.limit_stdout()
    return False


def benchmark_log(
    name: str,
    columns: list[str],
    *,
    out: TextIO | None = None,
    metadata: dict[str, Any] | None = None,
    start_runtime: bool = True,
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
        for a set of output csv files (one per rank) in that directory.
        (The legate command line option ` --benchmark-to-file` is equivalent to
        setting `LEGATE_BENCHMARK_OUT` to have the same value as the directory
        as `--logdir`.)
        For example, if `name` is `mybench`, then rank `P` will write its
        benchmark data to `mybench_[unique hex string].P.csv`.
    metadata: dict[str, Any] | None = None
        Optional dictionary of metadata that will be included in the header
        of the table.  If `None`, `legate.util.info.info()` will be used to
        generate the metadata.
    start_runtime: bool = True
        By default, `benchmark_log()` uses the legate runtime to populate
        metadata and, in multi-rank programs, to determine a globally unique id
        for each benchmark.  If `start_runtime=False` is specified, the legate
        runtime will not be started if it is not already running: each rank in
        a multiprocess program will generate a different id, and metadata about
        the runtime and machine configuration of legate will be missing.

    Returns
    -------
    BenchmarkLog | BenchmarkLogFromFilename
        A context manager whose one method is `log()`, which adds
        a row of benchmark data to the table.
    """
    uid = _benchmark_uid()
    num_nodes = 1
    node_id = 0

    use_runtime = start_runtime or runtime_has_started()
    if use_runtime:
        # local import to avoid starting the runtime unless requested
        from ._benchmark.id import _consensus_uid  # noqa: PLC0415

        num_nodes, node_id = _node_info()
        uid = _consensus_uid(uid)

    file = _benchmark_file(out)
    if metadata is None:
        metadata = cast(
            dict[str, Any], legate_info(start_runtime=start_runtime)
        )
    if file is not None:
        if _use_rich(file, num_nodes):
            return BenchmarkLogRich(name, uid, columns, file, metadata)
        return BenchmarkLogCSV(name, uid, columns, file, metadata)

    file_name = _benchmark_file_name(name, uid, node_id)

    def thunk(file: TextIO) -> BenchmarkLog:
        return BenchmarkLogCSV(name, uid, columns, file, metadata)

    return BenchmarkLogFromFilename(file_name, thunk)
