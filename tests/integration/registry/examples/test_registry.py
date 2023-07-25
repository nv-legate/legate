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
from registry import user_context, user_lib


def test_task_registrar():
    task = user_context.create_auto_task(user_lib.shared_object.HELLO)
    task.execute()


def test_task_immediate():
    task = user_context.create_auto_task(user_lib.shared_object.WORLD)
    task.execute()


def test_task_invalid():
    with pytest.raises(ValueError):
        user_context.create_auto_task(12345)

    with pytest.raises(ValueError):
        user_context.create_auto_task(user_lib.shared_object.NO_VARIANT)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
