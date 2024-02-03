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

from __future__ import annotations

import multiprocessing
import queue
import shlex
from datetime import datetime
from pathlib import Path
from typing import Any, NoReturn

from typing_extensions import Protocol

from ...util.colors import yellow
from ...util.types import ArgList, EnvDict
from ...util.ui import banner, summary
from .. import CUSTOM_FILES, PER_FILE_ARGS, FeatureType
from ..config import Config
from ..test_system import ProcessResult, TestSystem
from .util import Shard, StageResult, StageSpec, log_proc


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
    args: ArgList

    # --- Protocol methods

    def __init__(self, config: Config, system: TestSystem) -> None:
        ...

    def env(self, config: Config, system: TestSystem) -> EnvDict:
        """Generate stage-specific customizations to the process env

        Parameters
        ----------
        config: Config
            Test runner configuration

        system: TestSystem
            Process execution wrapper

        """
        ...

    def delay(self, shard: Shard, config: Config, system: TestSystem) -> None:
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

    def __call__(self, config: Config, system: TestSystem) -> None:
        """Execute this test stage.

        Parameters
        ----------
        config: Config
            Test runner configuration

        system: TestSystem
            Process execution wrapper

        """
        if config.other.gdb:
            self._launch_gdb(config, system)
            return

        t0 = datetime.now()
        procs = self._launch(config, system)
        t1 = datetime.now()

        self.result = StageResult(procs, t1 - t0)

    @property
    def name(self) -> str:
        """A stage name to display for tests in this stage."""
        return self.__class__.__name__

    @property
    def intro(self) -> str:
        """An informative banner to display at stage end."""
        workers = self.spec.workers
        workers_text = f"{workers} worker{'s' if workers > 1 else ''}"
        return (
            banner(f"Entering stage: {self.name} (with {workers_text})") + "\n"
        )

    @property
    def outro(self) -> str:
        """An informative banner to display at stage end."""
        total, passed = self.result.total, self.result.passed

        result = summary(self.name, total, passed, self.result.time)

        footer = banner(
            f"Exiting stage: {self.name}",
            details=(
                "* Results      : "
                + yellow(
                    f"{passed} / {total} files passed "  # noqa E500
                    f"({passed / total * 100:0.1f}%)"
                    if total > 0
                    else "0 tests are running, Please check "
                ),
                "* Elapsed time : " + yellow(f"{self.result.time}"),
            ),
        )

        return f"{result}\n{footer}"

    def file_args(self, test_file: Path, config: Config) -> ArgList:
        """Extra command line arguments based on the test file.

        Parameters
        ----------
        test_file : Path
            Path to a test file

        config: Config
            Test runner configuration

        """
        test_file_string = str(test_file)
        args = PER_FILE_ARGS.get(test_file_string, [])

        # These are a bit ugly but necessary in order to make pytest generate
        # more verbose output for integration tests when -v, -vv is specified
        if "integration" in test_file_string and config.info.verbose > 0:
            args += ["-v"]
        if "integration" in test_file_string and config.info.verbose > 1:
            args += ["-s"]

        return args

    def cov_args(self, config: Config) -> ArgList:
        """Coverage binary and coverage arguments.

        Parameters
        ----------
        config: Config
            Test runner configuration

        """
        if config.other.cov_bin:
            # By default legate will assume cov_bin is an executable instead
            # of a python script since it does not end in .py
            args = ["--run-mode=python"]

            args += [str(config.other.cov_bin)] + config.other.cov_args.split()
            if config.other.cov_src_path:
                args += ["--source", str(config.other.cov_src_path)]
        else:
            args = []

        return args

    def _run_common(
        self,
        cmd: ArgList,
        test_description: Path,
        config: Config,
        system: TestSystem,
        shard: Shard,
    ) -> ProcessResult:
        self.delay(shard, config, system)

        result = system.run(
            cmd,
            test_description,
            env=self._env(config, system),
            timeout=config.execution.timeout,
        )

        log_proc(self.name, result, config, verbose=config.info.verbose)

        self.shards.put(shard)

        return result

    def _run_gdb(
        self,
        cmd: ArgList,
        test_description: Path,
        config: Config,
        system: TestSystem,
        shard: Shard,
    ) -> NoReturn:
        import subprocess
        import sys

        subprocess.run(cmd, env=self._env(config, system))
        sys.exit()

    def run_python(
        self,
        test_file: Path,
        config: Config,
        system: TestSystem,
        *,
        custom_args: ArgList | None = None,
    ) -> ProcessResult:
        """Execute a single test files with appropriate environment and
        command-line options for a feature test stage.

        Parameters
        ----------
        test_file : Path
            Test file to execute

        config: Config
            Test runner configuration

        system: TestSystem
            Process execution wrapper

        """
        test_path = config.root_dir / test_file

        shard = self.shards.get()

        cov_args = self.cov_args(config)

        stage_args = self.args + self.shard_args(shard, config)
        if config.other.gdb:
            stage_args += ["--gdb"]

        file_args = self.file_args(test_file, config)

        cmd = (
            [str(config.legate_path)]
            + stage_args
            + cov_args
            # If both the python and Realm signal handlers are active, we may
            # not get good reporting of backtraces on crashes at the C++ level.
            # We are typically more interested in seeing the backtrace of the
            # crashing C++ thread, not the python code, so we ask pytest to not
            # install the python fault handler.
            + [str(test_path), "-p", "no:faulthandler"]
            + file_args
            + config.extra_args
        )

        if custom_args:
            cmd += custom_args

        if config.other.gdb:
            return self._run_gdb(cmd, Path(test_file), config, system, shard)
        else:
            return self._run_common(
                cmd, Path(test_file), config, system, shard
            )

    def run_gtest(
        self,
        test_file: Path,
        arg_test: str,
        config: Config,
        system: TestSystem,
        *,
        custom_args: ArgList | None = None,
    ) -> ProcessResult:
        """Execute a single test within gtest with appropriate environment and
        command-line options for a feature test stage.

        Parameters
        ----------
        test_file : str
            Test file to execute

        arg_test : str
            Test name to be executed

        config: Config
            Test runner configuration

        system: TestSystem
            Process execution wrapper

        """

        shard = self.shards.get()

        cov_args = self.cov_args(config)

        stage_args = self.args + self.shard_args(shard, config)
        if config.other.gdb:
            stage_args += ["--gdb"]

        file_args = self.file_args(test_file, config)

        cmd = (
            [str(config.legate_path)]
            + stage_args
            + cov_args
            + [str(test_file), f"--gtest_filter={arg_test}"]
            + file_args
            + config.extra_args
        )

        if custom_args:
            cmd += custom_args

        if config.other.gdb:
            cmd += ["--gtest_catch_exceptions=0"]

        if config.other.gdb:
            return self._run_gdb(cmd, Path(arg_test), config, system, shard)
        else:
            return self._run_common(cmd, Path(arg_test), config, system, shard)

    def run_mpi(
        self,
        test_file: Path,
        arg_test: str,
        config: Config,
        system: TestSystem,
        *,
        custom_args: ArgList | None = None,
    ) -> ProcessResult:
        """Execute a single test within gtest with appropriate environment and
        command-line options for a feature test stage.

        Parameters
        ----------
        test_file : str
            Test file to execute

        arg_test : str
            Test name to be executed

        config: Config
            Test runner configuration

        system: TestSystem
            Process execution wrapper

        """

        shard = self.shards.get()

        cov_args = self.cov_args(config)

        stage_args = self.args + self.shard_args(shard, config)
        file_args = self.file_args(test_file, config)

        launcher_args = (
            ["--launcher=mpirun"]
            + [f"--ranks-per-node={config.multi_node.ranks_per_node}"]
            + ["--launcher-extra=--merge-stderr-to-stdout"]
        )
        if filename := config.multi_node.mpi_output_filename:
            launcher_args += [
                '--launcher-extra="--output-filename"',
                f"--launcher-extra={shlex.quote(filename)}",
            ]

        cmd = (
            [str(config.legate_path)]
            + launcher_args
            + stage_args
            + cov_args
            + [str(test_file), f"--gtest_filter={arg_test}"]
            + file_args
            + config.extra_args
        )

        if custom_args:
            cmd += custom_args

        return self._run_common(cmd, Path(arg_test), config, system, shard)

    def _env(self, config: Config, system: TestSystem) -> EnvDict:
        env = dict(config.env)
        env.update(self.env(config, system))

        # special case for LEGATE_CONFIG -- if users have specified this on
        # their own we still want to see the value since it will affect the
        # test invocation directly.
        if "LEGATE_CONFIG" in system.env:
            env["LEGATE_CONFIG"] = system.env["LEGATE_CONFIG"]

        return env

    def _init(self, config: Config, system: TestSystem) -> None:
        self.spec = self.compute_spec(config, system)
        self.shards = system.manager.Queue(len(self.spec.shards))
        for shard in self.spec.shards:
            self.shards.put(shard)

    @staticmethod
    def _handle_multi_node_args(config: Config) -> ArgList:
        args: ArgList = []
        if config.multi_node.ranks_per_node > 1:
            args += [
                "--ranks-per-node",
                str(config.multi_node.ranks_per_node),
            ]
        if config.multi_node.nodes > 1:
            args += [
                "--nodes",
                str(config.multi_node.nodes),
            ]
        if config.multi_node.launcher != "none":
            args += ["--launcher", str(config.multi_node.launcher)]
        for extra in config.multi_node.launcher_extra:
            args += ["--launcher-extra=" + str(extra)]

        return args

    @staticmethod
    def _handle_cpu_pin_args(config: Config, shard: Shard) -> ArgList:
        args: ArgList = []
        if config.execution.cpu_pin != "none":
            args += [
                "--cpu-bind",
                str(shard),
            ]

        return args

    def _launch(
        self, config: Config, system: TestSystem
    ) -> list[ProcessResult]:
        pool = multiprocessing.pool.ThreadPool(self.spec.workers)

        assert not config.other.gdb

        if config.gtest_file:
            if config.multi_node.ranks_per_node > 1:
                jobs = [
                    pool.apply_async(
                        self.run_mpi, (config.gtest_file, arg, config, system)
                    )
                    for arg in config.gtest_tests
                ]
            else:
                jobs = [
                    pool.apply_async(
                        self.run_gtest,
                        (config.gtest_file, arg, config, system),
                    )
                    for arg in config.gtest_tests
                ]
        else:
            jobs = [
                pool.apply_async(self.run_python, (path, config, system))
                for path in config.test_files
            ]
        pool.close()

        sharded_results = [job.get() for job in jobs]

        custom = (x for x in CUSTOM_FILES if x.kind == self.kind)

        custom_results = [
            self.run_python(Path(x.file), config, system, custom_args=x.args)
            for x in custom
        ]

        return sharded_results + custom_results

    def _launch_gdb(self, config: Config, system: TestSystem) -> None:
        # with --gdb set the "run" functions below will exit

        if config.multi_node.ranks_per_node > 1 or config.multi_node.nodes > 1:
            raise ValueError("--gdb can only be used with a single rank")

        if config.gtest_file and config.gtest_tests:
            if len(config.gtest_tests) != 1:
                raise ValueError("--gdb can only be used with a single test")
            self.run_gtest(
                config.gtest_file, config.gtest_tests[0], config, system
            )

        else:
            if len(config.test_files) != 1:
                raise ValueError("--gdb can only be used with a single test")
            self.run_python(config.test_files[0], config, system)
