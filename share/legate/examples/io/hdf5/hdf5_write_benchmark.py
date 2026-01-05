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
from legate.timing import time

if TYPE_CHECKING:
    from argparse import Namespace


BIG_BANNER = "=" * 80


class BenchmarkResult(NamedTuple):
    """Results from a single benchmark write operation."""

    wall_time: float
    legate_time: float
    mb: float
    legate_throughput: float


class AggregatedResult(NamedTuple):
    """Aggregated results from multiple benchmark iterations."""

    shape: tuple[int, ...]
    dtype: str
    mb: float
    avg_wall_time: float
    avg_legate_time: float
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
        default=3,
        help="number of iterations per configuration",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing output directory",
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

    # We use the wall time to measure the time taken by the local process.
    # We use the legate time to measure the time taken by the Legate runtime
    # to complete the write operation across all ranks.
    wall_start = pytime.time()
    legate_start_us = time(units="us")

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
    legate_time = (time(units="us") - legate_start_us) / 1e6

    # Calculate total number of elements
    total_elements = math.prod(shape)

    mb_written = total_elements * legate_dtype.size / (1024 * 1024)
    legate_throughput = mb_written / legate_time if legate_time > 0 else 0

    return BenchmarkResult(
        wall_time=wall_time,
        legate_time=legate_time,
        mb=mb_written,
        legate_throughput=legate_throughput,
    )


def benchmark_configuration(
    shape: tuple[int, ...], dtype_str: str, output_dir: Path, iterations: int
) -> AggregatedResult:
    """
    Benchmark a specific shape/dtype configuration.

    Parameters
    ----------
    shape : tuple[int, ...]
        Array shape to benchmark.
    dtype_str : str
        Data type to benchmark.
    output_dir : Path
        Directory for output files.
    iterations : int
        Number of iterations to run.

    Returns
    -------
    AggregatedResult
        Named tuple containing aggregated benchmark results.
    """
    shape_str = "x".join(f"{dim:,}" for dim in shape)
    print(f"Benchmarking shape={shape_str}, dtype={dtype_str}")

    iter_results = []
    for iteration in range(iterations):
        output_file = (
            output_dir
            / f"benchmark_{shape_str}_{dtype_str}_iter{iteration}.h5"
        )

        result = benchmark_write(output_file, shape, dtype_str)
        iter_results.append(result)

        print(
            f"  Iteration {iteration + 1}: "
            f"Wall={result.wall_time:.3f}s, "
            f"Legate={result.legate_time:.3f}s, "
            f"Throughput={result.legate_throughput:.2f} MB/s"
        )

    # Calculate averages
    avg_wall = sum(r.wall_time for r in iter_results) / len(iter_results)
    avg_legate = sum(r.legate_time for r in iter_results) / len(iter_results)
    avg_throughput = sum(r.legate_throughput for r in iter_results) / len(
        iter_results
    )
    mb_written = iter_results[0].mb

    print(
        f"  Average: Wall={avg_wall:.3f}s, "
        f"Legate={avg_legate:.3f}s, "
        f"Throughput={avg_throughput:.2f} MB/s"
    )
    print()

    return AggregatedResult(
        shape=shape,
        dtype=dtype_str,
        mb=mb_written,
        avg_wall_time=avg_wall,
        avg_legate_time=avg_legate,
        avg_throughput=avg_throughput,
    )


def run_benchmarks(args: Namespace) -> None:
    """Run all benchmark configurations."""
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(BIG_BANNER)
    print("HDF5 WRITE BENCHMARK")
    print(BIG_BANNER)
    print(f"Output directory: {args.output_dir}")

    shape_str = f"({','.join(str(dim) for dim in args.shape)})"

    print(f"Shape: {shape_str}")
    print(f"Data types: {args.dtypes}")
    print(f"Iterations per config: {args.iterations}")
    print(BIG_BANNER)
    print()

    results = []

    for dtype_str in args.dtypes:
        result = benchmark_configuration(
            args.shape, dtype_str, args.output_dir, args.iterations
        )
        results.append(result)

    print(BIG_BANNER)
    print("BENCHMARK SUMMARY")
    print(BIG_BANNER)
    print(
        f"{'Shape':>15} {'Type':>8} {'MB':>10} {'Wall(s)':>10} "
        f"{'Legate(s)':>10} {'Throughput':>12}"
    )
    print("-" * 85)

    for r in results:
        shape_str = "x".join(f"{dim:,}" for dim in r.shape)
        print(
            f"{shape_str:>15} {r.dtype:>8} {r.mb:10.2f} "
            f"{r.avg_wall_time:10.3f} {r.avg_legate_time:10.3f} "
            f"{r.avg_throughput:10.2f} MB/s"
        )

    print(BIG_BANNER)

    # Find best performance
    if results:
        best = max(results, key=lambda x: x.avg_throughput)
        shape_str = "x".join(f"{dim:,}" for dim in best.shape)

        print(
            f"\nBest throughput: {best.avg_throughput:.2f} MB/s "
            f"(shape={shape_str}, dtype={best.dtype})"
        )


def main() -> None:
    """Main benchmark function."""
    args = parse_arguments()

    run_benchmarks(args)
    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
