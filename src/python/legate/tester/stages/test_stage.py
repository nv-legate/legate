# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, NoReturn, Protocol

from rich.console import Group
from rich.rule import Rule

from ...util.ui import section, summary
from ..defaults import FEATURES, PROCESS_ENV
from ..runner import Runner, TestSpec
from .util import MANUAL_CONFIG_ENV, Shard, StageResult, StageSpec, log_proc

if TYPE_CHECKING:
    import queue

    from rich.panel import Panel

    from ...util.types import ArgList, EnvDict
    from .. import FeatureType
    from ..config import Config
    from ..test_system import ProcessResult, TestSystem


class TestStage(Protocol):
    """Encapsulate running configured test files using specific features.

    Parameters
    ----------
    config: Config
        Test runner configuration

    system: TestSystem
        Process execution wrapper

    """

    kind: FeatureType

    #: The computed specification for processes to launch to run the
    #: configured test files.
    spec: StageSpec

    #: The computed sharding id sets to use for job runs
    shards: queue.Queue[Any]

    #: After the stage completes, results will be stored here
    result: StageResult

    #: Any fixed stage-specific command-line args to pass
    args: ClassVar[ArgList] = []

    runner: Runner

    # --- Protocol methods

    def __init__(self, config: Config, system: TestSystem) -> None:
        pass

    def stage_env(self, config: Config, system: TestSystem) -> EnvDict:
        """Generate stage-specific customizations to the process env.

        Parameters
        ----------
        config: Config
            Test runner configuration

        system: TestSystem
            Process execution wrapper

        """
        ...

    def delay(
        self,
        shard: Shard,  # noqa: ARG002
        config: Config,  # noqa: ARG002
        system: TestSystem,  # noqa: ARG002
    ) -> None:
        """Wait any delay that should be applied before running the next
        test.

        Parameters
        ----------
        shard: Shard
            The shard to be used for the next test that is run

        config: Config
            Test runner configuration

        system: TestSystem
            Process execution wrapper

        """
        return

    def shard_args(self, shard: Shard, config: Config) -> ArgList:
        """Generate the command line arguments necessary to launch
        the next test process on the given shard.

        Parameters
        ----------
        shard: Shard
            The shard to be used for the next test that is run

        config: Config
            Test runner configuration

        """
        ...

    def compute_spec(self, config: Config, system: TestSystem) -> StageSpec:
        """Compute the number of worker processes to launch and stage shards
        to use for running the configured test files.

        Parameters
        ----------
        config: Config
            Test runner configuration

        system: TestSystem
            Process execution wrapper

        """
        ...

    # --- Shared implementation methods

    def execute(self, config: Config, system: TestSystem) -> None:
        """Execute this test stage.

        Parameters
        ----------
        config: Config
            Test runner configuration

        system: TestSystem
            Process execution wrapper

        """
        if config.other.gdb:
            self._launch_gdb_and_exit(config, system)

        t0 = datetime.now()
        procs = self._launch(config, system)
        t1 = datetime.now()

        self.result = StageResult(procs, t1 - t0)

    def env(self, config: Config, system: TestSystem) -> EnvDict:
        """Compute the environment variables to set for this test execution,
        taking into account any per-stage or per-project customizations
        into account.

        Parameters
        ----------
        config: Config
            Test runner configuration

        system: TestSystem
            Process execution wrapper

        Returns
        -------
            EnvDict

        """
        # start with general env values needed for every legate test
        env = dict(PROCESS_ENV)

        # can't use auto-resource configuration in any tests
        env.update(MANUAL_CONFIG_ENV)

        # Do this here so that stage_env() and project.stage_env() can
        # potentially override it.
        if config.other.color:
            env.setdefault("FORCE_COLOR", "1")

        # add general stage environment customizations
        env.update(self.stage_env(config, system))

        # add project-specific stage environment customizations
        env.update(config.project.stage_env(self.kind))

        # special case for LEGATE_CONFIG -- if users have specified this on
        # their own we still want to see the value since it will affect the
        # test invocation directly.
        if "LEGATE_CONFIG" in system.env:
            env["LEGATE_CONFIG"] = system.env["LEGATE_CONFIG"]

        return env

    @property
    def name(self) -> str:
        """A stage name to display for tests in this stage."""
        return self.__class__.__name__

    @property
    def intro(self) -> Panel:
        """An informative banner to display at stage end."""
        num_workers = self.spec.workers
        workers = f"with {num_workers} worker{'s' if num_workers > 1 else ''}"
        return section(f"Entering stage: [cyan]{self.name}[/] ({workers})")

    @property
    def outro(self) -> Group:
        """An informative banner to display at stage end."""
        total, passed = self.result.total, self.result.passed

        return Group(
            section(
                Group(
                    f"Exiting stage: [cyan]{self.name}[/]\n",
                    summary(total, passed, self.result.time),
                )
            ),
            Rule(style="white"),
        )

    def _run(
        self,
        test_spec: TestSpec,
        config: Config,
        system: TestSystem,
        *,
        custom_args: ArgList | None = None,
    ) -> ProcessResult:
        """Execute a single test within gtest with appropriate environment and
        command-line options for a feature test stage.

        Parameters
        ----------
        test_spec : TestSpec
            Specification for the test to execute

        config: Config
            Test runner configuration

        system: TestSystem
            Process execution wrapper

        """
        shard = self.shards.get()

        stage_args = self.args + self.shard_args(shard, config)

        cmd = self.runner.cmd(
            test_spec, config, stage_args, custom_args=custom_args
        )

        self.delay(shard, config, system)

        result = system.run(
            cmd,
            test_spec.display,
            env=self.env(config, system),
            timeout=config.execution.timeout,
        )

        log_proc(self.name, result, config, verbose=config.info.verbose)

        self.shards.put(shard)

        return result

    def _init(self, config: Config, system: TestSystem) -> None:
        self.runner = Runner.create(config)
        self.spec = self.compute_spec(config, system)
        self.shards = system.manager.Queue(len(self.spec.shards))
        for shard in self.spec.shards:
            self.shards.put(shard)

    @staticmethod
    def handle_multi_node_args(config: Config) -> ArgList:  # noqa: D102
        args: ArgList = []

        if config.multi_node.launcher != "none":
            args += ["--launcher", str(config.multi_node.launcher)]

        if config.multi_node.ranks_per_node > 1:
            args += ["--ranks-per-node", str(config.multi_node.ranks_per_node)]

        if config.multi_node.nodes > 1:
            args += ["--nodes", str(config.multi_node.nodes)]

        for extra in config.multi_node.launcher_extra:
            args += ["--launcher-extra=" + str(extra)]

        return args

    @staticmethod
    def handle_cpu_pin_args(  # noqa: D102
        config: Config, shard: Shard
    ) -> ArgList:
        args: ArgList = []
        if config.execution.cpu_pin != "none":
            args += ["--cpu-bind", str(shard)]

        return args

    def _launch(
        self, config: Config, system: TestSystem
    ) -> list[ProcessResult]:
        pool = ThreadPool(self.spec.workers)

        assert not config.other.gdb

        custom_paths_map = {
            Path(x.file): x for x in config.project.custom_files()
        }

        specs = self.runner.test_specs(config)

        sharded_specs = (s for s in specs if s.path not in custom_paths_map)
        jobs = [
            pool.apply_async(self._run, (spec, config, system))
            for spec in sharded_specs
        ]
        pool.close()

        sharded_results = [job.get() for job in jobs]

        unsharded_specs = [s for s in specs if s.path in custom_paths_map]
        unsharded_results = []
        for spec in unsharded_specs:
            kind = custom_paths_map[spec.path].kind or FEATURES
            args = custom_paths_map[spec.path].args or []
            if self.kind == kind or self.kind in kind:
                result = self._run(spec, config, system, custom_args=args)
                unsharded_results.append(result)

        return sharded_results + unsharded_results

    def _launch_gdb_and_exit(
        self, config: Config, system: TestSystem
    ) -> NoReturn:
        import os
        import sys
        from subprocess import run

        cmd = self.runner.cmd_gdb(config)
        env = os.environ | self.env(config, system)

        run(cmd, env=env, check=False)

        sys.exit()
