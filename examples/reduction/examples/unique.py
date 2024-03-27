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
from reduction import unique

import legate.core.types as ty
from legate.core import get_legate_runtime


def test(n: int, radix: int, print_stores: bool):
    # Generate inputs using cuNumeric
    input = get_legate_runtime().create_store(ty.int32, (n,))
    np.asarray(input)[:] = np.random.randint(
        low=0, high=10, size=n, dtype="int32"
    )

    if print_stores:
        print(np.asarray(input))

    result = unique(input, radix=radix)
    if print_stores:
        print(np.asarray(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        type=int,
        default=100,
        dest="n",
        help="Number of elements in the input store",
    )
    parser.add_argument(
        "-r",
        "--radix",
        type=int,
        default=4,
        dest="radix",
        help="Fan-in of the reduction tree",
    )
    parser.add_argument(
        "--print-stores",
        default=False,
        dest="print_stores",
        action="store_true",
        help="Print stores",
    )
    args, _ = parser.parse_known_args()

    test(args.n, args.radix, args.print_stores)
