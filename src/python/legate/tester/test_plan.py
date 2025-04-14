# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provide a TestPlan class to coordinate multiple feature test stages."""

from __future__ import annotations

from datetime import timedelta
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.rule import Rule
from rich.text import Text

from ..util.ui import banner, section, summary, table, warn
from . import LAST_FAILED_FILENAME
from .logger import LOG
from .stages import STAGES, log_proc

if TYPE_CHECKING:
    from rich.panel import Panel

    from .config import Config
    from .test_system import ProcessResult, TestSystem

__all__ = ("TestPlan",)


class TestPlan:
    """Encapsulate an entire test run with multiple feature test stages.

    Parameters
    ----------
    config: Config
        Test runner configuration

    system: TestSystem
        Process execution wrapper

    """

    def __init__(self, config: Config, system: TestSystem) -> None:
        self._config = config
        self._system = system
        self._stages = [
            STAGES[feature](config, system) for feature in config.features
        ]

    def execute(self) -> int:
        """Execute the entire test run with all configured feature stages."""
        # This code path will exit the process
        if self._config.other.gdb:
            if len(self._stages) != 1:
                msg = "--gdb only works with a single stage"
                raise ValueError(msg)
            self._stages[0].execute(self._config, self._system)

        LOG.clear()

        LOG(self.intro)

        for stage in self._stages:
            LOG(stage.intro)
            stage.execute(self._config, self._system)
            LOG(stage.outro)

        all_procs = tuple(
            chain.from_iterable(s.result.procs for s in self._stages)
        )
        total = len(all_procs)
        passed = sum(proc.passed for proc in all_procs)

        self._log_failures(all_procs)

        self._record_last_failed(all_procs)

        LOG(self.outro(total, passed))

        return int((total - passed) > 0)

    @property
    def intro(self) -> Panel:
        """An informative banner to display at test run start."""
        cpus = len(self._system.cpus)
        try:
            gpus: int | str = len(self._system.gpus)
        except RuntimeError:
            gpus = "N/A"

        details: dict[str, Any] = {
            "Feature stages": ", ".join(x for x in self._config.features),
            "System description": f"{cpus} cpus / {gpus} gpus",
        }

        if self._config.gtest_tests:
            details["Test files per stage"] = len(self._config.gtest_tests)
            ranks = self._config.multi_node.ranks_per_node
            kind = "OpenMPI" if ranks > 1 else "GTest"
            title = f"Test Suite Configuration ({kind})"
        else:
            details["Test files per stage"] = len(self._config.test_files)
            title = "Test Suite Configuration (Python)"

        return banner(title, content=table(details, quote=False))

    def outro(self, total: int, passed: int) -> Panel:
        """An informative banner to display at test run end.

        Parameters
        ----------
        total: int
            Number of total tests that ran in all stages

        passed: int
            Number of tests that passed in all stages

        """
        details = (
            f"* {s.name: <6}: {s.result.passed} / {s.result.total} passed in "
            f"{s.result.time.total_seconds():0.2f}s"
            for s in self._stages
        )

        content = Text.from_markup("\n".join(details) + "\n\n")

        time = sum((s.result.time for s in self._stages), timedelta(0, 0))
        content += summary(total, passed, time)

        return banner("Overall summary", content=content)

    def _log_failures(self, all_procs: tuple[ProcessResult, ...]) -> None:
        if all(proc.passed for proc in all_procs):
            return

        LOG(section("FAILURES"))

        for stage in self._stages:
            procs = (proc for proc in stage.result.procs if not proc.passed)
            for proc in procs:
                log_proc(stage.name, proc, self._config, verbose=True)

        LOG(Rule(style="dim white"))

    def _record_last_failed(
        self, all_procs: tuple[ProcessResult, ...]
    ) -> None:
        fails = {proc.test_display for proc in all_procs if not proc.passed}

        if not fails:
            return

        try:
            with Path(LAST_FAILED_FILENAME).open(mode="w") as f:
                f.write("\n".join(sorted(str(x) for x in fails)))
        except OSError:
            warn("Couldn't write last-fails")
