# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from legate.core import LocalTaskID, get_legate_runtime, types as ty


class Test_scalar_arg:
    def test_scalar_arg_with_array_type(self) -> None:
        runtime = get_legate_runtime()
        # Create a task object only to test validation logic
        task = runtime.create_auto_task(runtime.core_library, LocalTaskID(1))
        with pytest.raises(ValueError):  # noqa: PT011
            task.add_scalar_arg(1, ty.array_type(ty.int8, 1))

    def test_array_size_mismatch(self) -> None:
        runtime = get_legate_runtime()
        # Create a task object only to test validation logic
        task = runtime.create_auto_task(runtime.core_library, LocalTaskID(1))
        with pytest.raises(ValueError):  # noqa: PT011
            task.add_scalar_arg((1, 2, 3), ty.array_type(ty.int8, 1))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
