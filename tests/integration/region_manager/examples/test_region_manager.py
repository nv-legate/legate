# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import pytest
from region_manager import user_context, user_lib

import legate.core.types as ty


def test_region_manager():
    task = user_context.create_auto_task(user_lib.cffi.TESTER)
    for _ in range(2000):
        store = user_context.create_store(ty.int64, shape=(10,))
        task.add_output(store)
    task.execute()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
