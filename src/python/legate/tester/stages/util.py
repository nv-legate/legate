# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

from ...util.ui import failed, passed, shell, skipped, timeout
from ..logger import LOG

if TYPE_CHECKING:
    from datetime import datetime, timedelta

    from ..config import Config
    from ..test_system import ProcessResult

MANUAL_CONFIG_ENV = {"LEGATE_AUTO_CONFIG": "0"}

UNPIN_ENV = {"REALM_SYNTHETIC_CORE_MAP": ""}

RankShard: TypeAlias = tuple[int, ...]


@dataclass(frozen=True)
class Shard:
    """Specify how resources should be allotted for each test process."""

    #: A list of shards for each rank
    ranks: list[RankShard]

    def __str__(self) -> str:  # noqa: D105
        return "/".join(",".join(str(r) for r in rank) for rank in self.ranks)


@dataclass(frozen=True)
class StageSpec:
    """Specify the operation of a test run."""

    #: The number of worker processes to start for running tests
    workers: int

    # A list of (cpu or gpu) shardings to draw on for each test
    shards: list[Shard]


@dataclass(frozen=True)
class StageResult:
    """Collect results from all tests in a TestStage."""

    #: Individual test process results including return code and stdout.
    procs: list[ProcessResult]

    #: Cumulative execution time for all tests in a stage.
    time: timedelta

    @property
    def total(self) -> int:
        """The total number of tests run in this stage."""
        return len(self.procs)

    @property
    def passed(self) -> int:
        """The number of tests in this stage that passed."""
        return sum(p.passed for p in self.procs)


def adjust_workers(
    workers: int, requested_workers: int | None, *, detail: str | None = None
) -> int:
    """Adjust computed workers according to command line requested workers.

    The final number of workers will only be adjusted down by this function.

    Parameters
    ----------
    workers: int
        The computed number of workers to use

    requested_workers: int | None, optional
        Requested number of workers from the user, if supplied (default: None)

    detail: str | None, optional
        Additional information to provide in case the adjusted number of
        workers is zero (default: None)

    Returns
    -------
    int
        The number of workers to actually use

    """
    if requested_workers is not None and requested_workers < 0:
        msg = "requested workers must be non-negative"
        raise ValueError(msg)

    if requested_workers == 0:
        msg = "requested workers must not be zero"
        raise RuntimeError(msg)

    if requested_workers is not None:
        if requested_workers > workers:
            msg = (
                f"Requested workers ({requested_workers}) is greater than "
                f"computed workers ({workers})"
            )
            raise RuntimeError(msg)
        workers = requested_workers

    if workers == 0:
        msg = "Current configuration results in zero workers"
        if detail:
            msg += f" [details: {detail}]"
        raise RuntimeError(msg)

    return workers


def format_duration(start: datetime, end: datetime) -> str:
    r"""Format a duration from START to END for display.

    Parameters
    ----------
    start : datetime
        The start of the duration
    end : datetime
        The end of the duration

    Returns
    -------
    str
        The formatted duration

    Raises
    ------
    ValueError
        If the duration is invalid, such as when end comes before start.
    """
    if end < start:
        msg = f"End ({end}) happens before start ({start})"
        raise ValueError(msg)

    duration = (end - start).total_seconds()
    time = f"{duration:0.2f}s"
    start_str = start.strftime("%H:%M:%S.%f")[:-4]
    end_str = end.strftime("%H:%M:%S.%f")[:-4]
    return f" [yellow]{time}[/] [dim]{{{start_str}, {end_str}}}[/]"


def log_proc(
    name: str, proc: ProcessResult, config: Config, *, verbose: bool
) -> None:
    """Log a process result according to the current configuration."""
    if config.info.debug or config.dry_run:
        LOG("\n")
        LOG(shell(proc.invocation))

    if proc.time is None or proc.start is None or proc.end is None:
        duration = ""
    else:
        assert proc.end - proc.start == proc.time
        duration = format_duration(proc.start, proc.end)

    msg = f"({name}){duration} {proc.test_display}"
    details = proc.output.split("\n") if verbose else None
    if proc.skipped:
        LOG(skipped(msg))
    elif proc.timeout:
        LOG(timeout(msg, details=details))
    elif proc.returncode == 0:
        LOG(passed(msg, details=details))
    else:
        LOG(failed(msg, details=details, exit_code=proc.returncode))
