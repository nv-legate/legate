# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provide a System class to encapsulate process execution and reporting
system information (number of CPUs present, etc).

"""

from __future__ import annotations

import os
import multiprocessing
from dataclasses import dataclass
from datetime import datetime, timedelta
from signal import SIGINT
from subprocess import PIPE, STDOUT, Popen, TimeoutExpired
from typing import TYPE_CHECKING

import psutil

from ..util.system import System

if TYPE_CHECKING:
    from collections.abc import Sequence

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

    #: User-friendly string to use in reported output
    test_display: str

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

    def __init__(self, *, dry_run: bool = False) -> None:
        super().__init__()
        self.manager = multiprocessing.Manager()
        self.dry_run: bool = dry_run

    @property
    def memory(self) -> int:  # noqa: D102
        return psutil.virtual_memory().total

    def run(
        self,
        cmd: Sequence[str],
        test_display: str,
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

        test_display : str
            User-friendly string to use in reported output

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
            return ProcessResult(invocation, test_display, skipped=True)

        full_env = dict(os.environ)
        full_env.update(env)

        start = datetime.now()
        with Popen(
            cmd,
            cwd=cwd,
            env=full_env,
            stdout=PIPE,
            stderr=STDOUT,
            errors="replace",
        ) as proc:
            try:
                # wait on process to exit via communicate, don't use
                # wait() due to potential deadlock with pipes
                output, _ = proc.communicate(timeout=timeout)
            except TimeoutExpired:
                os.kill(proc.pid, SIGINT)
                output, _ = proc.communicate()

                assert timeout is not None  # mypy
                assert output is not None  # mypy
                return ProcessResult(
                    invocation,
                    test_display,
                    start=start,
                    end=start + timedelta(seconds=timeout),
                    timeout=True,
                    output=output,
                )

            end = datetime.now()
            assert proc.stdout is not None  # mypy
            return ProcessResult(
                invocation,
                test_display,
                start=start,
                end=end,
                returncode=proc.returncode,
                output=output,
            )
