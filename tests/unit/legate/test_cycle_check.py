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


import os
import subprocess
from pathlib import Path

import pytest

PROG_TEXT = """
import numpy as np
from legate.core import get_legate_runtime, types as ty
store = get_legate_runtime().core_context.create_store(
    ty.int32, shape=(4,), optimize_scalar=False
)
# initialize the RegionField backing the store
store.storage
# create a cycle
x = [store]
x.append(x)
"""


def test_cycle_check(tmp_path: Path) -> None:
    prog_file = tmp_path / "prog.py"
    prog_file.write_text(PROG_TEXT)
    env = os.environ.copy()
    env["LEGATE_CYCLE_CHECK"] = "1"
    output = subprocess.check_output(
        [
            "legate",
            prog_file,
            "--cpus",
            "1",
        ],
        env=env,
    )
    assert "found cycle!" in output.decode("utf-8")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
