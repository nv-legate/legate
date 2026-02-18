# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import sys
import shutil
import tempfile
import contextlib
from importlib.util import find_spec
from pathlib import Path
from subprocess import PIPE, STDOUT, CalledProcessError, CompletedProcess, run
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Generator


def subprocess_helper(
    file: str,
    case: str,
    env: dict[str, str],
    check: bool = True,
    faulthandler: bool = True,
    legate_config: str = "",
) -> CompletedProcess[bytes]:
    """Helper for launching test case in a new legate process."""
    cov_args = []
    pytest_args = ["-sv"]
    if bool(os.environ.get("COVERAGE_RUN")) and find_spec("coverage"):
        cov_args = ["coverage", "run", "-m"]
    if not faulthandler:
        pytest_args += ["-p", "no:faulthandler"]

    pruned_env = os.environ.copy()
    del pruned_env["REALM_BACKTRACE"]
    # Always force single CPU to avoid insufficient resource in subproc
    pruned_env["LEGATE_AUTO_CONFIG"] = "0"
    pruned_env["LEGATE_CONFIG"] = "--cpus 1 " + legate_config
    pruned_env["SUBPROCESS_TEST"] = "1"
    pruned_env.update(env)

    try:
        p = run(
            [
                sys.executable,
                "-m",
                *cov_args,
                "pytest",
                *pytest_args,
                f"{file}::{case}",
            ],
            stdout=PIPE,
            stderr=STDOUT,
            check=check,
            env=pruned_env,
        )
    except CalledProcessError as exc:
        # otherwise we don't get the actual failure
        raise RuntimeError(exc.stdout.decode())
    return p


@pytest.fixture
def run_subprocess() -> Callable[..., CompletedProcess[Any]] | None:
    if os.getenv("SUBPROCESS_TEST"):
        return None
    return subprocess_helper


@pytest.fixture
def tmp_path(request: pytest.FixtureRequest) -> Generator[Path, None, None]:
    """Provide a temporary directory path consistent across all ranks.

    This overrides pytest's built-in `tmp_path` fixture, which creates a
    unique directory per process. Instead, this fixture creates a
    deterministic path based on the test name. This ensures all MPI ranks
    use the same directory in multi-rank tests.

    The directory is created before the test and cleaned up after.
    """
    # Use the full node ID (e.g. "tests/test_foo.py::TestClass::test_bar")
    # to guarantee uniqueness across different test files and classes.
    test_id = request.node.nodeid
    # Sanitize for filesystem (replace special chars)
    safe_name = "".join(c if c.isalnum() else "_" for c in test_id)

    # tempfile.gettempdir() respects TMPDIR; set it to namespace
    # by run if concurrent tests are running.
    base_dir = Path(tempfile.gettempdir()) / "legate_tests"
    shared_path = base_dir / safe_name

    # Create the directory (all ranks do this, but it's idempotent)
    shared_path.mkdir(parents=True, exist_ok=True)

    try:
        yield shared_path
    finally:
        # Ensure Legate has finished any I/O under shared_path before cleanup.
        fence_error = None
        try:
            from legate.core import get_legate_runtime  # noqa: PLC0415

            get_legate_runtime().issue_execution_fence(block=True)
        except Exception as e:
            fence_error = e
        # Cleanup after the test (all ranks attempt this).
        # Only suppress FileNotFoundError since another rank may have
        # already removed the directory.
        with contextlib.suppress(FileNotFoundError):
            shutil.rmtree(shared_path)
        if fence_error is not None:
            raise fence_error
