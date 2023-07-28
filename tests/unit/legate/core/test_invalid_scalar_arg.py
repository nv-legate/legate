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


import pytest

from legate.core import get_legate_runtime, types as ty


class Test_scalar_arg:
    def test_unimplemented_types(self) -> None:
        context = get_legate_runtime().core_context
        # Create a task object only to test validation logic
        task = context.create_auto_task(1)
        with pytest.raises(NotImplementedError):
            task.add_scalar_arg(None, ty.struct_type([ty.int8]))
        with pytest.raises(NotImplementedError):
            task.add_scalar_arg(
                (1,), ty.array_type(ty.struct_type([ty.int8]), 1)
            )
        with pytest.raises(NotImplementedError):
            task.add_scalar_arg((1,), (ty.struct_type([ty.int8]),))

    def test_scalar_arg_with_array_type(self) -> None:
        context = get_legate_runtime().core_context
        # Create a task object only to test validation logic
        task = context.create_auto_task(1)
        with pytest.raises(ValueError):
            task.add_scalar_arg(1, ty.array_type(ty.int8, 1))

    def test_array_size_mismatch(self) -> None:
        context = get_legate_runtime().core_context
        # Create a task object only to test validation logic
        task = context.create_auto_task(1)
        with pytest.raises(ValueError):
            task.add_scalar_arg((1, 2, 3), ty.array_type(ty.int8, 1))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
