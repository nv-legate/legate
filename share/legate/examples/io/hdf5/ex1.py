#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

# ruff: noqa: T201
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import TYPE_CHECKING

import h5py

from legate.core.io.hdf5 import from_file
from legate.timing import time

if TYPE_CHECKING:
    from argparse import Namespace
    from collections.abc import Iterator


def parse_arguments() -> Namespace:
    """
    Parse command-line arguments using argparse.

    Returns
    -------
    Parsed arguments containing filename and number of ranks.
    """
    parser = ArgumentParser(
        description="Read datasets from HDF5 files using Legate."
    )
    parser.add_argument(
        "filename", type=Path, help="prefix for the HDF5 file names"
    )
    parser.add_argument(
        "--n_rank",
        type=int,
        default=1,
        metavar="int",
        help="number of ranks (files)",
    )
    args = parser.parse_args()

    if not args.filename.exists():
        parser.error(f"File path {args.filename} does not exist")
    if not args.filename.is_file():
        parser.error(f"File path {args.filename} must be a readable file")

    if args.n_rank <= 0:
        parser.error(f"Number of ranks must be > 0 (have {args.n_rank})")

    return args


def print_io_mode() -> None:
    r"""Print the I/O mode based on the environment variable
    'LEGATE_IO_USE_VFD_GDS'.
    """
    match os.environ.get("LEGATE_IO_USE_VFD_GDS", "").casefold():
        case "1":
            print("IO MODE : GDS")
        case _:
            print("IO MODE : POSIX")


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


def process_hdf5_files(filename: Path, n_rank: int) -> None:
    r"""Read HDF5 datasets simultaneously, and compute throughput. The
    datasets are virtual datasets stored across 8 files. Here, each rank opens
    the top level file and recurse through all the datasets and read them
    simultaneously. Please refer to the hdf5 data generator program to create
    such a dataset.

    Parameters
    ----------
    filename: Path
        Path to the toplevel HDF5 file.
    n_rank: int
        Number of ranks.
    """
    total_size = 0
    # Use a legate timer instead of regular Python timers. Legate timers will
    # only capture the work performed inside Legate tasks (and will do so
    # asynchronously), while Python timers will capture all of the below.
    start_time_us = time(units="us")
    fname = str(filename)
    for _ in range(n_rank):
        with h5py.File(fname, "r") as hdf_file:
            for dset in traverse_datasets(hdf_file):
                data = from_file(fname, dataset_name=dset)
                total_size += data.size

    elapsed_time_s = (time(units="us") - start_time_us) * 10**6
    throughput = total_size / (
        elapsed_time_s * (2**20)
    )  # Throughput in MB/sec

    print(f"Total Data Read: {total_size}")
    print(f"Total Turnaround Time (seconds): {elapsed_time_s}")
    print(f"Throughput (MB/sec): {throughput}")


def main() -> None:
    """
    Main function to perform reading of hdf5 virtual datasets in
    rank-per-GPU mode where each rank will be executed on a specific GPU.
    """
    args = parse_arguments()
    print_io_mode()
    process_hdf5_files(args.filename, args.n_rank)


if __name__ == "__main__":
    main()
