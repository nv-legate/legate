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
from reduction import bincount, categorize, histogram

import legate.core.types as ty
from legate.core import get_legate_runtime


def test(size: int, num_bins: int):
    input = get_legate_runtime().create_store(ty.int32, (size,))
    np.asarray(input)[:] = np.random.randint(
        low=0, high=size - 1, size=size, dtype="int32"
    )
    bins = get_legate_runtime().create_store(ty.int32, (num_bins + 1,))
    np.asarray(bins)[:] = np.arange(num_bins + 1) * size // num_bins
    print("Input:")
    print(np.asarray(input))
    print("Bin edges:")
    print(np.asarray(bins))

    tmp = categorize(input, bins)
    result = bincount(tmp, num_bins)
    print("Histogram via bincount:")
    print(np.asarray(result))

    result = histogram(input, bins)
    print("Direct histogram implementation:")
    print(np.asarray(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        type=int,
        default=10,
        dest="size",
        help="Number of elements",
    )
    parser.add_argument(
        "-b",
        type=int,
        default=3,
        dest="num_bins",
        help="Number of bins",
    )
    args, _ = parser.parse_known_args()

    test(args.size, args.num_bins)
