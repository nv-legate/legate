#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import time as pytime
from argparse import ArgumentParser
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import h5py

from legate.core import get_legate_runtime
from legate.io.hdf5 import from_file
from legate.util.benchmark import benchmark_log
from legate.util.info import info as legate_info

if TYPE_CHECKING:
    from argparse import Namespace
    from collections.abc import Iterator


class ReadResult(NamedTuple):
    """Results from a single benchmark read operation."""

    wall_time: float
    total_bytes: int
    mb: float
    throughput_mb_s: float


def parse_arguments() -> Namespace:
    """
    Parse command-line arguments using argparse.

    Returns
    -------
    Parsed arguments containing filename and benchmark options.
    """
    parser = ArgumentParser(
        description="Benchmark HDF5 read performance using Legate."
    )
    parser.add_argument(
        "filename", type=Path, help="prefix for the HDF5 file names"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="number of iterations to run (default: 3)",
    )
    parser.add_argument(
        "--max-throughput",
        type=float,
        default=None,
        metavar="MB/s",
        help="max theoretical throughput of the cluster in MB/s; "
        "stored as metadata for comparison with benchmark results",
    )
    args = parser.parse_args()

    if not args.filename.exists():
        parser.error(f"File path {args.filename} does not exist")
    if not args.filename.is_file():
        parser.error(f"File path {args.filename} must be a readable file")

    return args


def traverse_datasets(hdf_file: h5py.File) -> Iterator[str]:
    r"""
    Recursively traverse datasets in an HDF5 file.

    Parameters
    ----------
    hdf_file: h5py.File
        Open HDF5 file object.

    Yields
    ------
    Path to each dataset.
    """

    def h5py_dataset_iterator(
        group: h5py.File | h5py.Group, prefix: str = ""
    ) -> Iterator[str]:
        for key, item in group.items():
            path = f"{prefix}/{key}"
            if isinstance(item, h5py.Dataset):  # Check if it is a dataset
                yield path
            elif isinstance(item, h5py.Group):  # Check if it is a group
                yield from h5py_dataset_iterator(item, prefix=path)

    yield from h5py_dataset_iterator(hdf_file)


def read_hdf5_once(filename: Path) -> ReadResult:
    """Read HDF5 datasets once and return timing results.

    Parameters
    ----------
    filename: Path
        Path to the toplevel HDF5 file.

    Returns
    -------
    ReadResult
        Named tuple containing timing and throughput metrics.
    """
    runtime = get_legate_runtime()
    total_size = 0

    # Start wall time measurement
    wall_start = pytime.time()

    fname = str(filename)
    with h5py.File(fname, "r") as hdf_file:
        for dset in traverse_datasets(hdf_file):
            data = from_file(fname, dataset_name=dset)
            total_size += data.size * data.type.size

    # Block to ensure all operations complete
    runtime.issue_execution_fence(block=True)

    wall_time = pytime.time() - wall_start

    mb_read = total_size / (1024 * 1024)
    throughput = mb_read / wall_time if wall_time > 0 else 0

    return ReadResult(
        wall_time=wall_time,
        total_bytes=total_size,
        mb=mb_read,
        throughput_mb_s=throughput,
    )


def process_hdf5_files(
    filename: Path, iterations: int, max_throughput_mb_s: float | None = None
) -> None:
    r"""Read HDF5 datasets and benchmark throughput using the legate benchmark
    framework. Opens the top-level file and recurses through all datasets,
    reading them and logging timing/throughput.

    Parameters
    ----------
    filename: Path
        Path to the toplevel HDF5 file.
    iterations: int
        Number of benchmark iterations.
    max_throughput_mb_s: float, optional
        Max theoretical throughput of the cluster in MB/s; stored as metadata
        for comparison with benchmark results.
    """
    # Define columns for the benchmark log
    columns = [
        "iteration",
        "total_bytes",
        "mb",
        "wall_time_s",
        "throughput_mb_s",
    ]

    # Additional metadata specific to this benchmark
    metadata = {
        "Benchmark Config": {"File": str(filename), "Iterations": iterations},
        **legate_info(),
    }
    if max_throughput_mb_s is not None:
        metadata["Max throughput (MB/s)"] = f"{max_throughput_mb_s:.2f}"

    results: list[ReadResult] = []

    with benchmark_log(
        "hdf5_read", columns=columns, metadata=metadata
    ) as blog:
        for iteration in range(iterations):
            result = read_hdf5_once(filename)
            results.append(result)

            blog.log(
                iteration=iteration,
                total_bytes=result.total_bytes,
                mb=f"{result.mb:.2f}",
                wall_time_s=f"{result.wall_time:.3f}",
                throughput_mb_s=f"{result.throughput_mb_s:.2f}",
            )


def main() -> None:
    """
    Main function to benchmark reading of HDF5 virtual datasets.
    Each rank will read datasets and timing/throughput will be logged
    using the legate benchmark framework.
    """
    args = parse_arguments()
    process_hdf5_files(
        args.filename, args.iterations, max_throughput_mb_s=args.max_throughput
    )


if __name__ == "__main__":
    main()
