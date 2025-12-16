# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import sys
from importlib.util import find_spec
from subprocess import PIPE, STDOUT, CalledProcessError, CompletedProcess, run
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable


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
