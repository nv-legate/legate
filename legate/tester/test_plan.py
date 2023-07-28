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

"""Provide a TestPlan class to coordinate multiple feature test stages.

"""
from __future__ import annotations

from datetime import timedelta
from itertools import chain

from ..util.colors import yellow
from ..util.ui import banner, rule, summary, warn
from . import LAST_FAILED_FILENAME
from .config import Config
from .logger import LOG
from .stages import STAGES, log_proc
from .test_system import ProcessResult, TestSystem


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
        LOG.clear()

        LOG(self.intro)

        for stage in self._stages:
            LOG(stage.intro)
            stage(self._config, self._system)
            LOG(stage.outro)

        all_procs = tuple(
            chain.from_iterable(s.result.procs for s in self._stages)
        )
        total = len(all_procs)
        passed = sum(proc.passed for proc in all_procs)

        LOG(f"\n{rule(pad=4)}")

        self._log_failures(all_procs)

        self._log_timeouts(all_procs)

        self._record_last_failed(all_procs)

        LOG(self.outro(total, passed))

        return int((total - passed) > 0)

    @property
    def intro(self) -> str:
        """An informative banner to display at test run start."""

        cpus = len(self._system.cpus)
        try:
            gpus: int | str = len(self._system.gpus)
        except RuntimeError:
            gpus = "N/A"

        details = (
            f"* Feature stages       : {', '.join(yellow(x) for x in self._config.features)}",  # noqa E501
            f"* Test files per stage : {yellow(str(len(self._config.test_files)))}",  # noqa E501
            f"* TestSystem description   : {yellow(str(cpus) + ' cpus')} / {yellow(str(gpus) + ' gpus')}",  # noqa E501
        )
        return banner("Test Suite Configuration", details=details)

    def outro(self, total: int, passed: int) -> str:
        """An informative banner to display at test run end.

        Parameters
        ----------
        total: int
            Number of total tests that ran in all stages

        passed: int
            Number of tests that passed in all stages

        """
        details = [
            f"* {s.name: <6}: "
            + yellow(
                f"{s.result.passed} / {s.result.total} passed in {s.result.time.total_seconds():0.2f}s"  # noqa E501
            )
            for s in self._stages
        ]

        time = sum((s.result.time for s in self._stages), timedelta(0, 0))
        details.append("")
        details.append(
            summary("All tests", total, passed, time, justify=False)
        )

        overall = banner("Overall summary", details=details)

        return f"{overall}\n"

    def _log_failures(self, all_procs: tuple[ProcessResult, ...]) -> None:
        if not any(proc.returncode for proc in all_procs):
            return

        LOG(f"{banner('FAILURES')}\n")

        for stage in self._stages:
            procs = (proc for proc in stage.result.procs if proc.returncode)
            for proc in procs:
                log_proc(stage.name, proc, self._config, verbose=True)

    def _log_timeouts(self, all_procs: tuple[ProcessResult, ...]) -> None:
        if not any(proc.timeout for proc in all_procs):
            return

        LOG(f"{banner('TIMEOUTS')}\n")

        for stage in self._stages:
            procs = (proc for proc in stage.result.procs if proc.timeout)
            for proc in procs:
                log_proc(stage.name, proc, self._config, verbose=True)

    def _record_last_failed(
        self, all_procs: tuple[ProcessResult, ...]
    ) -> None:
        fails = {proc.test_file for proc in all_procs if not proc.passed}

        if not fails:
            return

        try:
            with open(LAST_FAILED_FILENAME, "w") as f:
                f.write("\n".join(sorted(str(x) for x in fails)))
        except OSError:
            warn("Couldn't write last-fails")
