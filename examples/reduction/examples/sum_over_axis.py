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


import cunumeric as np
from reduction import sum_over_axis

import legate.core.types as ty
from legate.core import get_legate_runtime


def test():
    store = get_legate_runtime().create_store(ty.int64, (4, 5))
    np.asarray(store).fill(1)
    print(np.asarray(store))

    result1 = sum_over_axis(store, 0)
    print(np.asarray(result1))

    result2 = sum_over_axis(store, 1)
    print(np.asarray(result2))


if __name__ == "__main__":
    test()
