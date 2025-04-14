# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Consolidate test configuration from command-line and environment."""

from __future__ import annotations

import os
import sys
import json
import shlex
import shutil
import operator
import functools
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich import reconfigure

from ..util.types import (
    ArgList,
    DataclassMixin,
    LauncherType,
    object_to_dataclass,
)
from . import LAST_FAILED_FILENAME, FeatureType, defaults
from .args import PinOptionsType, parser

if TYPE_CHECKING:
    from argparse import Namespace

    from .project import Project

__all__ = ("Config",)


@dataclass(frozen=True)
class Core(DataclassMixin):
    cpus: int
    gpus: int
    omps: int
    ompthreads: int
    utility: int


@dataclass(frozen=True)
class Memory(DataclassMixin):
    sysmem: int
    fbmem: int
    numamem: int


@dataclass(frozen=True)
class MultiNode(DataclassMixin):
    nodes: int
    ranks_per_node: int
    launcher: LauncherType
    launcher_extra: list[str]
    mpi_output_filename: str | None

    def __post_init__(self, **kw: dict[str, Any]) -> None:
        # fix up launcher_extra to automatically handle quoted strings with
        # internal whitespace, have to use __setattr__ for frozen
        # https://docs.python.org/3/library/dataclasses.html#frozen-instances
        if self.launcher_extra:
            ex: list[str] = functools.reduce(
                operator.iadd,
                (shlex.split(x) for x in self.launcher_extra),
                [],
            )
            object.__setattr__(self, "launcher_extra", ex)


@dataclass(frozen=True)
class Execution(DataclassMixin):
    workers: int | None
    timeout: int | None
    bloat_factor: int
    gpu_delay: int
    cpu_pin: PinOptionsType


@dataclass(frozen=True)
class Info(DataclassMixin):
    verbose: bool
    debug: bool


@dataclass
class Other(DataclassMixin):
    dry_run: bool
    gdb: bool
    cov_bin: str | None
    cov_args: str
    cov_src_path: str | None
    color: bool

    # not frozen because we have to update this manually
    legate_install_dir: Path | None


class Config:
    """A centralized configuration object that provides the information
    needed by test stages in order to run.

    Parameters
    ----------
    argv : ArgList
        command-line arguments to use when building the configuration

    """

    def __init__(self, argv: ArgList, *, project: Project) -> None:
        self.argv = argv
        self.project = project

        args, self._extra_args = parser.parse_known_args(self.argv[1:])
        args.gtest_skip_list = set(args.gtest_skip_list)
        # only saving this for help with testing
        self._args = args

        # feature configuration
        self.features = self._compute_features(args)

        self.core = object_to_dataclass(args, Core)
        self.memory = object_to_dataclass(args, Memory)
        self.multi_node = object_to_dataclass(args, MultiNode)
        self.execution = object_to_dataclass(args, Execution)
        self.info = object_to_dataclass(args, Info)
        self.other = object_to_dataclass(args, Other)
        self.other.legate_install_dir = self._compute_legate_install_dir(args)

        # test selection
        self.examples = not args.cov_bin
        self.integration = True
        self.files = args.files
        self.last_failed = args.last_failed
        self.test_root = args.test_root
        # NOTE: This reads the rest of the configuration, so do it last
        self.gtest_tests = self._compute_gtest_tests(args)

        if self.multi_node.nodes > 1 and self.multi_node.launcher == "none":
            msg = (
                "Requested multi-node configuration with "
                f"--nodes {self.multi_node.nodes} but did not specify a "
                "launcher. Must use --launcher to specify a launcher."
            )
            raise RuntimeError(msg)

        if (
            self.multi_node.ranks_per_node > 1
            and self.multi_node.launcher == "none"
        ):
            msg = (
                "Requested multi-rank configuration with "
                f"--ranks-per-node {self.multi_node.ranks_per_node} but did "
                "not specify a launcher. Must use --launcher to specify a "
                "launcher."
            )
            raise RuntimeError(msg)

        color_system = "auto" if self.other.color else None
        reconfigure(soft_wrap=True, color_system=color_system)

    @property
    def dry_run(self) -> bool:
        """Whether a dry run is configured."""
        return self.other.dry_run

    @property
    def extra_args(self) -> ArgList:
        """Extra command-line arguments to pass on to individual test files."""
        return self._extra_args

    @property
    def root_dir(self) -> Path:
        """Path to the directory containing the tests."""
        if self.test_root:
            return Path(self.test_root)

        # if not explicitly given, just use cwd assuming we are at a repo top
        return Path.cwd()

    @property
    def test_files(self) -> tuple[Path, ...]:
        """List of all test files to use for each stage.

        An explicit list of files from the command line will take precedence.

        Otherwise, the files are computed based on command-line options, etc.

        """
        if self.files and self.last_failed:
            msg = "Cannot specify both --files and --last-failed"
            raise RuntimeError(msg)

        if self.files is not None:
            return self.files

        if self.last_failed and (last_failed := self._read_last_failed()):
            return last_failed

        files = []

        if self.examples:
            examples = (
                path.relative_to(self.root_dir)
                for path in self.root_dir.joinpath("examples").glob("*.py")
                if str(path.relative_to(self.root_dir))
                not in self.project.skipped_examples()
            )
            files.extend(sorted(examples))

        if self.integration:
            integration_tests = (
                path.relative_to(self.root_dir)
                for path in self.root_dir.joinpath("tests/integration").glob(
                    "*.py"
                )
            )
            files.extend(sorted(integration_tests))

        return tuple(files)

    @property
    def legate_path(self) -> str:
        """Computed path to the legate driver script."""
        if not hasattr(self, "legate_path_"):

            def compute_legate_path() -> str:
                if self.other.legate_install_dir is not None:
                    return str(
                        self.other.legate_install_dir / "bin" / "legate"
                    )

                if legate_bin := shutil.which("legate"):
                    return legate_bin

                return str(
                    Path(__file__).resolve().parent.parent
                    / "driver"
                    / "driver_exec.py"
                )

            self.legate_path_ = compute_legate_path()
        return self.legate_path_

    def _compute_features(self, args: Namespace) -> tuple[FeatureType, ...]:
        computed: list[FeatureType]
        if args.features is not None:
            computed = args.features
        else:
            computed = [
                feature
                for feature in defaults.FEATURES
                if os.environ.get(f"USE_{feature.upper()}", None) == "1"
            ]

        # if nothing is specified any other way, at least run CPU stage
        if len(computed) == 0:
            computed.append("cpus")

        return tuple(computed)

    def _compute_legate_install_dir(self, args: Namespace) -> Path | None:
        # self._legate_source below is purely for testing
        if args.legate_install_dir:
            self._legate_source = "cmd"
            return Path(args.legate_install_dir)
        if "LEGATE_INSTALL_DIR" in os.environ:
            self._legate_source = "env"
            return Path(os.environ["LEGATE_INSTALL_DIR"])
        self._legate_source = "install"
        return None

    def _read_last_failed(self) -> tuple[Path, ...]:
        try:
            with Path(LAST_FAILED_FILENAME).open() as f:
                lines = (line.strip() for line in f)
                return tuple(Path(line) for line in lines if line)
        except OSError:
            return ()

    @staticmethod
    def _get_cached_tests(
        gtest_file: Path, cache_file: Path
    ) -> list[str] | None:
        if not cache_file.exists():
            return None

        # If the gtest_file is newer than the cache file, ignore it
        if gtest_file.stat().st_mtime > cache_file.stat().st_mtime:
            return None

        with cache_file.open() as fd:
            return json.load(fd)

    @staticmethod
    def _cache_tests(cache_file: Path, test_names: list[str]) -> None:
        with cache_file.open("w") as fd:
            json.dump(test_names, fd)

    def _compute_gtest_tests_single(
        self,
        gtest_file: Path | str,
        args: Namespace,  # noqa: ARG002
    ) -> list[str]:
        gtest_file = Path(gtest_file).resolve()
        if not gtest_file.exists():
            msg = f"gtest binary: '{gtest_file}' does not appear to exist"
            raise ValueError(msg)

        cache_file = gtest_file.parent / (gtest_file.name + ".cached_gtests")
        if cached_tests := self._get_cached_tests(gtest_file, cache_file):
            return cached_tests

        env = os.environ.copy()
        env["LEGATE_AUTO_CONFIG"] = "0"
        # LSAN prints the suppressions that were applied at the end of the run
        # by default, e.g.
        #
        # -----------------------------------------------------
        # Suppressions used:
        #   count      bytes template
        #   3             50 librealm.*
        # -----------------------------------------------------
        #
        # Which trips up our parsing of the tests because we incorrectly label
        # "Suppressions used" as a test group, and "count bytes template", "3
        # 50 librealm" as a test.
        env["LSAN_OPTIONS"] = (
            env.get("LSAN_OPTIONS", "") + ":print_suppressions=0:"
        )
        try:
            cmd_out = subprocess.check_output(
                [str(gtest_file), "--gtest_list_tests"],
                stderr=subprocess.STDOUT,
                env=env,
            )
        except subprocess.CalledProcessError as cpe:
            print("Failed to fetch GTest tests")  # noqa: T201
            if cpe.stdout:
                print(f"stdout:\n{cpe.stdout.decode()}")  # noqa: T201
            if cpe.stderr:
                print(f"stderr:\n{cpe.stderr.decode()}")  # noqa: T201
            raise

        result = cmd_out.decode(sys.stdout.encoding).splitlines()

        test_group = ""
        test_names = []
        for line in result:
            # Skip empty entry
            if not line.strip():
                continue

            # Check if this is a test group
            if line[0] != " ":
                test_group = line.split("#")[0].strip()
                continue

            test_name = test_group + line.split("#")[0].strip()
            # Assign test to test group
            test_names.append(test_name)

        self._cache_tests(cache_file, test_names)
        return test_names

    def _filter_gtests(
        self, test_names: list[str], args: Namespace
    ) -> list[str]:
        skip_list = args.gtest_skip_list
        gtest_filter = args.gtest_filter
        is_parallel = (
            self.multi_node.ranks_per_node > 1 or self.multi_node.nodes > 1
        )

        def is_death_test(name: str) -> bool:
            return name.split(".", maxsplit=1)[0].endswith("DeathTest")

        def keep_test(name: str) -> bool:
            if name in skip_list:
                return False

            # Skip death tests when running with multiple processes. It looks
            # as if GTest catches the failure and declares the test successful,
            # but for some reason the failure is not actually completely
            # neutralized, and the exit code is non-zero.
            if is_parallel and is_death_test(name):
                return False

            return gtest_filter.match(name)

        if skip_list or is_parallel or (gtest_filter.pattern != ".*"):
            test_names = [name for name in test_names if keep_test(name)]
        return test_names

    def _compute_gtest_tests(self, args: Namespace) -> dict[Path, list[str]]:
        if args.gtest_files is None:
            return {}

        all_tests: dict[Path, list[str]] = {}
        for gtest_file in args.gtest_files:
            raw_test_names = self._compute_gtest_tests_single(gtest_file, args)
            all_tests[gtest_file] = self._filter_gtests(raw_test_names, args)

        if args.gtest_tests:
            remain_tests = {
                t for test in args.gtest_tests for t in test.split()
            }

            for test_file in list(all_tests.keys()):
                test_list = [
                    t for t in all_tests[test_file] if t in remain_tests
                ]
                if test_list:
                    all_tests[test_file] = test_list
                else:
                    del all_tests[test_file]

        return all_tests
