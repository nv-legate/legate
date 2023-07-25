# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import cunumeric as np
from reduction import bincount, user_context

import legate.core.types as ty


def test():
    size = 100
    num_bins = 10

    # Generate random inputs using cuNumeric
    store = user_context.create_store(ty.uint64, size)
    np.asarray(store)[:] = np.random.randint(
        low=0, high=num_bins - 1, size=size
    )
    print(np.asarray(store))

    result = bincount(store, num_bins)
    print(np.asarray(result))


if __name__ == "__main__":
    test()
