# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Consolidate test configuration from command-line and environment."""

from __future__ import annotations

import sys
import shlex
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from ..util.types import ArgList, Command
    from .config import Config


@dataclass(frozen=True)
class TestSpec:
    """A specification for a single test to execute, including a test file,
    any extra argument to identify the specific test, and a display value
    to use in logs.

    """

    path: Path
    display: str
    arg: str | None = None


class Runner:
    """Encapsulate how to run test commands for different kinds of execution
    scenarios.

    """

    @classmethod
    def create(cls, config: Config) -> Runner:
        """Create a specific Runner subclass suitable for the given
        configuration.

        Parameters
        ----------
        config: Config
            Test runner configuration

        Returns
        -------
            Runner

        """
        # TODO: change this when legate/gtest subcommands split
        if config.gtest_tests:
            return GTestRunner()
        return LegateRunner()

    def cmd_gdb(self, config: Config) -> Command:
        """Generate the command invocation to run a test under gdb.

        Parameters
        ----------
        config: Config
            Test runner configuration

        Returns
        -------
            Command

        """
        raise NotImplementedError

    def cmd(
        self,
        test_spec: TestSpec,
        config: Config,
        stage_args: ArgList,
        *,
        custom_args: ArgList | None = None,
    ) -> Command:
        """Generate the command invocation to run a single test.

        Parameters
        ----------
        test_spec: TestSpec
            Specification for the test command to generate

        config: Config
            Test runner configuration

        stage_args : ArgList
            Args specific to the stage, e.g. sharding, provided by TestStage

        Returns
        -------
            Command

        """
        raise NotImplementedError

    def test_specs(self, config: Config) -> tuple[TestSpec, ...]:
        """Generate a specification for each test to run.

        The specification includes the path to the test, a display value
        for the test in logs, and any additional arguments necessary to
        to identify or execute the test.

        Parameters
        ----------
        config : Config
            Test runner configuration

        """
        raise NotImplementedError


class LegateRunner(Runner):
    def cmd_gdb(self, config: Config) -> Command:
        """Generate the command invocation to run a single legate python test
        under gdb.

        Parameters
        ----------
        config: Config
            Test runner configuration

        Returns
        -------
            Command

        """
        if config.multi_node.ranks_per_node > 1 or config.multi_node.nodes > 1:
            msg = "--gdb can only be used with a single rank"
            raise ValueError(msg)

        if len(config.test_files) == 0:
            msg = "--gdb can only be used with a single test (none were given)"
            raise ValueError(msg)

        if len(config.test_files) > 1:
            msg = "--gdb can only be used with a single test"
            raise ValueError(msg)

        spec = TestSpec(config.test_files[0], str(config.test_files[0]))
        return self.cmd(spec, config, [])

    def cmd(
        self,
        test_spec: TestSpec,
        config: Config,
        stage_args: ArgList,
        *,
        custom_args: ArgList | None = None,
    ) -> Command:
        """Generate the command invocation to run a single legate python test.

        Parameters
        ----------
        test_spec: TestSpec
            Specification for the test command to generate

        config: Config
            Test runner configuration

        stage_args : ArgList
            Args specific to the stage, e.g. sharding, provided by TestStage

        Returns
        -------
            Command

        """
        gdb_args = ["--gdb"] if config.other.gdb else []
        cov_args = self.cov_args(config)
        file_args = self.file_args(test_spec.path, config)

        # If both Python and Realm signal handlers are active, we may not get
        # good backtraces on crashes at the C++ level. We are typically more
        # interested in seeing the backtrace of the crashing C++ thread, not
        # the python code, so we ask pytest to omit the python fault handler.
        pytest_args = ["-p", "no:faulthandler"]

        cmd = [
            sys.executable,
            str(config.legate_path),
            *stage_args,
            *gdb_args,
            *cov_args,
            str(config.root_dir / test_spec.path),
            *pytest_args,
            *file_args,
            *config.extra_args,
        ]

        if custom_args:
            cmd += custom_args

        return tuple(cmd)

    def test_specs(self, config: Config) -> tuple[TestSpec, ...]:
        """Generate extra args that go along with the test file.

        Test args are also what will be identified in the logs. For the
        Legate runner, this is the only purpose they are used for.

        Parameters
        ----------
        config : Config
            Test runner configuration

        """
        return tuple(
            TestSpec(path=path, display=str(path))
            for path in config.test_files
        )

    def file_args(self, test_file: Path, config: Config) -> ArgList:
        """Generate extra command line arguments based on the test file.

        Parameters
        ----------
        test_file : Path
            Path to a test file

        config: Config
            Test runner configuration

        """
        test_file_string = str(test_file)
        args = []

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

            args += [str(config.other.cov_bin), *config.other.cov_args.split()]
            if config.other.cov_src_path:
                args += ["--source", str(config.other.cov_src_path)]
        else:
            args = []

        return args


class GTestRunner(Runner):
    def cmd_gdb(self, config: Config) -> Command:
        """Generate the command invocation to run a gtest test under gdb.

        Parameters
        ----------
        config: Config
            Test runner configuration

        Returns
        -------
            Command

        """
        if config.multi_node.ranks_per_node > 1 or config.multi_node.nodes > 1:
            msg = "--gdb can only be used with a single rank"
            raise ValueError(msg)

        gtest_tests = config.gtest_tests
        # Remove test binaries that have empty test lists (since they didn't
        # meet some search criteria)
        filtered = {k: v for k, v in gtest_tests.items() if v}

        match sum(map(len, filtered.values())):
            case 0:
                msg = (
                    "--gdb can only be used with a single test (none were "
                    "given)"
                )
                raise ValueError(msg)
            case 1:
                pass
            case _:
                msg = "--gdb can only be used with a single test"
                raise ValueError(msg)

        test_bin = next(iter(filtered.keys()))
        spec = TestSpec(test_bin, "", filtered[test_bin][0])
        return self._cmd_single(spec, config, [])

    def cmd(
        self,
        test_spec: TestSpec,
        config: Config,
        stage_args: ArgList,
        *,
        custom_args: ArgList | None = None,
    ) -> Command:
        """

        Parameters
        ----------
        test_spec : TestSpec
            Specification for the test command to generate

        config: Config
            Test runner configuration

        stage_args : ArgList
            Args specific to the stage, e.g. sharding, provided by TestStage

        Returns
        -------
            Command

        """
        multi_rank = (
            config.multi_node.ranks_per_node > 1 or config.multi_node.nodes > 1
        )

        func = self._cmd_multi if multi_rank else self._cmd_single

        return func(test_spec, config, stage_args, custom_args=custom_args)

    def test_specs(self, config: Config) -> tuple[TestSpec, ...]:
        """Generate extra args that go along with the test file.

        Test args are also what will be identified in the logs. For the
        GTest runner, the test args are the filter that identifies a
        specific test case in the one gtest test file to run.

        Parameters
        ----------
        config : Config
            Test runner configuration

        """
        return tuple(
            TestSpec(path=test_file, arg=test, display=test)
            for test_file, test_list in config.gtest_tests.items()
            for test in test_list
        )

    def gtest_args(
        self, test_spec: TestSpec, *, gdb: bool = False, color: bool = False
    ) -> ArgList:
        """Generate args specific to configuring gtest.

        Parameters
        ----------
        test_spec: TestSpec
            Specification for the test command to generate gtest args for

        gdb : bool
            whether gdb is active

        Returns
        -------
        args : ArgList

        """
        args = [str(test_spec.path), f"--gtest_filter={test_spec.arg}"]
        if color:
            args.append("--gtest_color=yes")
        if gdb:
            args.append("--gtest_catch_exceptions=0")
        return args

    def _cmd_single(
        self,
        test_spec: TestSpec,
        config: Config,
        stage_args: ArgList,
        *,
        custom_args: ArgList | None = None,
    ) -> Command:
        """Execute a single test within gtest with appropriate environment and
        command-line options for a feature test stage.

        Parameters
        ----------
        test_spec: TestSpec
            Specification for the test command to generate

        config: Config
            Test runner configuration

        """
        gdb_args = ["--gdb"] if config.other.gdb else []
        gtest_args = self.gtest_args(
            test_spec, gdb=config.other.gdb, color=config.other.color
        )

        cmd = [
            sys.executable,
            str(config.legate_path),
            *stage_args,
            *gdb_args,
            *gtest_args,
            *config.extra_args,
        ]

        if custom_args:
            cmd += custom_args

        return tuple(cmd)

    def _cmd_multi(
        self,
        test_spec: TestSpec,
        config: Config,
        stage_args: ArgList,
        *,
        custom_args: ArgList | None = None,
    ) -> Command:
        """Execute a single test within gtest with appropriate environment and
        command-line options for a feature test stage.

        Parameters
        ----------
        test_spec: TestSpec
            Specification for the test command to generate

        config: Config
            Test runner configuration

        """
        assert not config.other.gdb

        gtest_args = self.gtest_args(test_spec, color=config.other.color)

        cmd = [
            sys.executable,
            str(config.legate_path),
            *stage_args,
            *gtest_args,
            *config.extra_args,
        ]

        if config.multi_node.launcher == "mpirun":
            cmd += ["--launcher-extra=--merge-stderr-to-stdout"]

            if filename := config.multi_node.mpi_output_filename:
                cmd += [
                    '--launcher-extra="--output-filename"',
                    f"--launcher-extra={shlex.quote(str(filename))}",
                ]

        if custom_args:
            cmd += custom_args

        return tuple(cmd)
