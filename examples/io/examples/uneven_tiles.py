# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import argparse

import cunumeric as np
from legateio import IOArray, read_uneven_tiles

import legate.core as lg


def test(shape: tuple[int, ...], dataset_name: str, print_input: bool):
    print(f"Input shape: {shape}")

    runtime = lg.get_legate_runtime()

    # Use cuNumeric to generate a random array to dump to a dataset
    src = np.random.randint(low=1, high=9, size=shape).astype("int8")

    if print_input:
        print(src)

    # Construct an IOArray from the cuNumeric ndarray
    c1 = IOArray.from_legate_data_interface(src.__legate_data_interface__)

    # Dump the IOArray to a dataset of uneven tiles
    c1.to_uneven_tiles(dataset_name)

    # Issue an execution fence and block on it. This is necessary because
    # the Python code for read_uneven_tiles needs to see the header that
    # becomes available only when the first writer task finishes.
    runtime.issue_execution_fence(block=True)

    # Read the dataset into an IOArray
    c2 = read_uneven_tiles(dataset_name)
    assert np.array_equal(np.asarray(c2), src)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--shape",
        type=int,
        nargs="+",
        default=(5, 5),
        dest="shape",
        help="Data shape",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="test",
        dest="dataset",
        help="Dataset name",
    )
    parser.add_argument(
        "--print-input",
        default=False,
        dest="print_input",
        action="store_true",
        help="Print input",
    )
    args, _ = parser.parse_known_args()

    test(args.shape, args.dataset, args.print_input)
