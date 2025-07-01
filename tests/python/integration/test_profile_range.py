# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import sys
import tempfile
import subprocess
from pathlib import Path

import pytest

from legate.core import get_legate_runtime

from .utils import tasks

TEST_FILE = """\
from legate.core import ProfileRange, VariantCode
from legate.core.task import task

@task(variants=tuple(VariantCode))
def profile_range_task() -> None:
    with ProfileRange("foobarbaz"):
        pass

profile_range_task()
"""

ASAN = "LD_PRELOAD" in os.environ


class TestProfileRange:
    def test_create_auto_task(self) -> None:
        runtime = get_legate_runtime()
        tasks.profile_range_task()
        runtime.issue_execution_fence(block=True)

    def _test_output(self, env: dict[str, str]) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            modpath = Path(tmpdir) / "prtest.py"
            modpath.write_text(TEST_FILE)
            env["LEGATE_CONFIG"] = f"--profile --logdir {tmpdir}"
            env["LEGATE_AUTO_CONFIG"] = "0"
            subprocess.run(
                [sys.executable, modpath],
                capture_output=True,
                text=True,
                env=env,
                check=True,
            )
            outpath = Path(tmpdir) / "legate_0.prof"
            with outpath.open("rb") as f:
                out = f.read()
                assert b"foobarbaz" in out
                assert b"profile_range_task" in out

    @pytest.mark.skipif(ASAN, reason="ASAN is configured")
    def test_output_without_asan(self) -> None:
        self._test_output({})

    @pytest.mark.skipif(not ASAN, reason="ASAN is not configured")
    def test_output_with_asan(self) -> None:
        self._test_output(dict(os.environ))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
