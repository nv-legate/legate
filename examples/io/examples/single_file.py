# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from legateio import IOArray, read_file, read_file_parallel

import legate.core as lg


def test(n: int, filename: str):
    print(f"Input shape: ({n},)")

    runtime = lg.get_legate_runtime()

    # Use cuNumeric to generate an array to dump to a file
    src = np.arange(n).astype("int8")

    # Construct an IOArray from the cuNumeric ndarray. They are aliased to the
    # same store after this call.
    c1 = IOArray.from_legate_data_interface(src.__legate_data_interface__)

    # Dump the IOArray to a file
    c1.to_file(filename)

    # Issue an execution fence to make sure the writer task finishes before
    # any of the downstream tasks start.
    #
    # Unlike data dependencies on stores that the runtime discovers and
    # enforces, the producer-consumer relationship mediated by some file IO is
    # invisible to the runtime. Issuing a fence between the producer and
    # consumer is one way to make the relationship visible. Another way is to
    # add a proper data dependence between the two, for example, by making the
    # producer return a scalar output that is later consumed by the consumer.
    runtime.issue_execution_fence()

    # Read the file into a IOArray
    c2 = read_file(filename, lg.int8)
    # Convert the IOArray to a cuNumeric ndarray so we can use cuNumeric for
    # equality check
    arr = np.asarray(c2)
    assert np.array_equal(src, arr)

    # Read the file into a IOArray with a fixed degree of parallelism
    c3 = read_file_parallel(filename, lg.int8, parallelism=2)
    assert np.array_equal(src, np.asarray(c3))

    # Read the file into a IOArray with the library-chosen degree of
    # parallelism
    c4 = read_file_parallel(filename, lg.int8)
    assert np.array_equal(src, np.asarray(c4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=10,
        dest="num",
        help="Number of elements",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default="test.dat",
        dest="file",
        help="File name",
    )
    args, _ = parser.parse_known_args()

    test(args.num, args.file)
