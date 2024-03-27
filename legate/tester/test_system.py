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

"""Provide a System class to encapsulate process execution and reporting
system information (number of CPUs present, etc).

"""
from __future__ import annotations

import multiprocessing
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from subprocess import PIPE, STDOUT, TimeoutExpired, run as stdlib_run
from typing import Sequence

import psutil

from ..util.system import System
from ..util.types import EnvDict

__all__ = ("TestSystem",)


def _quote(s: str) -> str:
    if " " in s:
        return repr(s)
    return s


@dataclass
class ProcessResult:
    #: The command invovation, including relevant environment vars
    invocation: str

    #: User-friendly test file path to use in reported output
    test_file: Path

    #: The time the process started
    start: datetime | None = None

    #: The time the process ended
    end: datetime | None = None

    #: Whether this process was actually invoked
    skipped: bool = False

    #: Whether this process timed-out
    timeout: bool = False

    #: The returncode from the process
    returncode: int = 0

    #: The collected stdout and stderr output from the process
    output: str = ""

    @property
    def time(self) -> timedelta | None:
        if self.start is None or self.end is None:
            return None
        return self.end - self.start

    @property
    def passed(self) -> bool:
        return self.returncode == 0 and not self.timeout


class TestSystem(System):
    """A facade class for system-related functions.

    Parameters
    ----------
    dry_run : bool, optional
        If True, no commands will be executed, but a log of any commands
        submitted to ``run`` will be made. (default: False)

    """

    def __init__(
        self,
        *,
        dry_run: bool = False,
    ) -> None:
        super().__init__()
        self.manager = multiprocessing.Manager()
        self.dry_run: bool = dry_run

    @property
    def memory(self) -> int:
        return psutil.virtual_memory().total

    def run(
        self,
        cmd: Sequence[str],
        test_file: Path,
        *,
        env: EnvDict | None = None,
        cwd: str | None = None,
        timeout: int | None = None,
    ) -> ProcessResult:
        """Wrapper for subprocess.run that encapsulates logging.

        Parameters
        ----------
        cmd : sequence of str
            The command to run, split on whitespace into a sequence
            of strings

        test_file : Path
            User-friendly test file path to use in reported output

        env : dict[str, str] or None, optional, default: None
            Environment variables to apply when running the command

        cwd: str or None, optional, default: None
            A current working directory to pass to stdlib ``run``.

        """

        env = env or {}

        envstr = (
            " ".join(f"{k}={_quote(v)}" for k, v in env.items())
            + min(len(env), 1) * " "
        )

        invocation = envstr + " ".join(cmd)

        if self.dry_run:
            return ProcessResult(invocation, test_file, skipped=True)

        full_env = dict(os.environ)
        full_env.update(env)

        start = datetime.now()
        try:
            proc = stdlib_run(
                cmd,
                cwd=cwd,
                env=full_env,
                stdout=PIPE,
                stderr=STDOUT,
                text=True,
                timeout=timeout,
            )
        except TimeoutExpired:
            return ProcessResult(invocation, test_file, timeout=True)

        end = datetime.now()

        return ProcessResult(
            invocation,
            test_file,
            start=start,
            end=end,
            returncode=proc.returncode,
            output=proc.stdout,
        )
