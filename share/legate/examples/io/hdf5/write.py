#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: T201
from __future__ import annotations

import math
import time as pytime
from argparse import ArgumentParser
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from legate.core import get_legate_runtime
from legate.core.types import float32, float64, int32, int64
from legate.io.hdf5 import to_file
from legate.util.benchmark import benchmark_log
from legate.util.info import info as legate_info

if TYPE_CHECKING:
    from argparse import Namespace


BIG_BANNER = "=" * 80


class BenchmarkResult(NamedTuple):
    """Results from a single benchmark write operation."""

    wall_time: float
    mb: float
    throughput: float


class AggregatedResult(NamedTuple):
    """Aggregated results from multiple benchmark iterations."""

    shape: tuple[int, ...]
    dtype: str
    mb: float
    avg_wall_time: float
    avg_throughput: float


def parse_shape(shape_str: str) -> tuple[int, ...]:
    """Parse a shape string into a tuple of integers.

    Parameters
    ----------
    shape_str : str
        String representation of the shape.
        Accepts formats like:
        - "1000" -> (1000,)
        - "1000,2000" -> (1000, 2000)
        - "(1000,2000)" -> (1000, 2000)

    Returns
    -------
    tuple[int, ...]
        Tuple of integers representing the shape.

    """
    shape_str = shape_str.strip()
    if shape_str.startswith("(") and shape_str.endswith(")"):
        shape_str = shape_str[1:-1]

    try:
        dims = [int(dim.strip()) for dim in shape_str.split(",")]
        return tuple(dims)
    except ValueError as e:
        error = f"Invalid shape format '{shape_str}': {e}"
        raise ValueError(error) from e


def parse_arguments() -> Namespace:
    """Parse command-line arguments."""
    parser = ArgumentParser(
        description="Benchmark HDF5 write performance using Legate."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("hdf5_benchmark"),
        help="directory for output HDF5 files (default: hdf5_benchmark)",
    )
    parser.add_argument(
        "--shape",
        type=parse_shape,
        default=(1000,),
        help="array shape to benchmark (e.g., '1000' for 1D, "
        "'1000,2000' for 2D)",
    )
    parser.add_argument(
        "--dtypes",
        type=str,
        nargs="+",
        default=["int32", "float32", "float64"],
        choices=["float32", "float64", "int32", "int64"],
        help="data types to benchmark",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="number of iterations per configuration",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing output directory",
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

    if args.output_dir.exists() and not args.overwrite:
        parser.error(
            f"Output directory {args.output_dir} already exists. "
            "Use --overwrite to overwrite it."
        )

    return args


def benchmark_write(
    output_file: Path, shape: tuple[int, ...], dtype_str: str
) -> BenchmarkResult:
    """
    Benchmark a single HDF5 write operation.

    Parameters
    ----------
    output_file : Path
        Path to the output HDF5 file.
    shape : tuple[int, ...]
        Shape of the array to write.
    dtype_str : str
        Data type as string.

    Returns
    -------
    BenchmarkResult
        Named tuple containing timing and throughput metrics.
    """
    runtime = get_legate_runtime()

    # Map dtype string to Legate type
    dtype_map = {
        "float32": float32,
        "float64": float64,
        "int32": int32,
        "int64": int64,
    }

    legate_dtype = dtype_map[dtype_str]

    # Start wall time measurement
    wall_start = pytime.time()

    # Creates an array with the given type and shape and fill it with a
    # constant value of 1. This array is written to a HDF5 file with
    # the given name and dataset name. Legate will create a HDF5
    # virtual dataset on disk for the dataset.
    array = runtime.create_array(dtype=legate_dtype, shape=shape)

    runtime.issue_fill(array, 1)
    to_file(array=array, path=output_file, dataset_name="/data")

    # We need to block here to ensure that the write operation is completed
    # before we can measure the time taken.
    runtime.issue_execution_fence(block=True)

    wall_time = pytime.time() - wall_start

    # Calculate total number of elements
    total_elements = math.prod(shape)

    mb_written = total_elements * legate_dtype.size / (1024 * 1024)
    throughput = mb_written / wall_time if wall_time > 0 else 0

    return BenchmarkResult(
        wall_time=wall_time, mb=mb_written, throughput=throughput
    )


def run_benchmarks(args: Namespace) -> None:
    """Run all benchmark configurations."""
    args.output_dir.mkdir(parents=True, exist_ok=True)

    shape_str = "x".join(str(dim) for dim in args.shape)

    # Define columns for the benchmark log
    columns = ["dtype", "iteration", "mb", "wall_time_s", "throughput_mb_s"]

    # Additional metadata specific to this benchmark
    metadata = {
        "Benchmark Config": {
            "Shape": shape_str,
            "Data Types": ", ".join(args.dtypes),
            "Iterations": args.iterations,
            "Output Directory": str(args.output_dir),
        },
        **legate_info(),
    }
    if args.max_throughput is not None:
        metadata["Max throughput (MB/s)"] = f"{args.max_throughput:.2f}"

    results: list[AggregatedResult] = []

    with benchmark_log(
        f"hdf5_write_{shape_str}", columns=columns, metadata=metadata
    ) as blog:
        for dtype_str in args.dtypes:
            iter_results: list[BenchmarkResult] = []

            for iteration in range(args.iterations):
                safe_shape_str = "x".join(str(dim) for dim in args.shape)
                filename = (
                    f"benchmark_{safe_shape_str}_{dtype_str}_"
                    f"iter{iteration}.h5"
                )
                output_file = args.output_dir / filename

                result = benchmark_write(output_file, args.shape, dtype_str)
                iter_results.append(result)

                blog.log(
                    dtype=dtype_str,
                    iteration=iteration,
                    mb=f"{result.mb:.2f}",
                    wall_time_s=f"{result.wall_time:.3f}",
                    throughput_mb_s=f"{result.throughput:.2f}",
                )

            avg_wall = sum(r.wall_time for r in iter_results) / len(
                iter_results
            )
            avg_throughput = sum(r.throughput for r in iter_results) / len(
                iter_results
            )
            mb_written = iter_results[0].mb

            results.append(
                AggregatedResult(
                    shape=args.shape,
                    dtype=dtype_str,
                    mb=mb_written,
                    avg_wall_time=avg_wall,
                    avg_throughput=avg_throughput,
                )
            )


def main() -> None:
    """Main benchmark function."""
    args = parse_arguments()

    run_benchmarks(args)
    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
