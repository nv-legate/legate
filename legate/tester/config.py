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

"""Consolidate test configuration from command-line and environment.

"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..util import colors
from ..util.types import (
    ArgList,
    DataclassMixin,
    EnvDict,
    LauncherType,
    object_to_dataclass,
)
from . import (
    FEATURES,
    LAST_FAILED_FILENAME,
    SKIPPED_EXAMPLES,
    FeatureType,
    defaults,
)
from .args import PinOptionsType, parser


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
            ex: list[str] = sum(
                (shlex.split(x) for x in self.launcher_extra), []
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

    # not frozen because we have to update this manually
    legate_dir: Path | None


class Config:
    """A centralized configuration object that provides the information
    needed by test stages in order to run.

    Parameters
    ----------
    argv : ArgList
        command-line arguments to use when building the configuration

    """

    def __init__(self, argv: ArgList) -> None:
        self.argv = argv

        args, self._extra_args = parser.parse_known_args(self.argv[1:])

        # only saving this for help with testing
        self._args = args

        colors.ENABLED = args.color

        # feature configuration
        self.features = self._compute_features(args)

        self.core = object_to_dataclass(args, Core)
        self.memory = object_to_dataclass(args, Memory)
        self.multi_node = object_to_dataclass(args, MultiNode)
        self.execution = object_to_dataclass(args, Execution)
        self.info = object_to_dataclass(args, Info)
        self.other = object_to_dataclass(args, Other)
        self.other.legate_dir = self._compute_legate_dir(args)

        # test selection
        self.examples = False if args.cov_bin else True
        self.integration = True
        self.unit = args.unit
        self.files = args.files
        self.last_failed = args.last_failed
        self.gtest_file = args.gtest_file
        self.test_root = args.test_root
        # NOTE: This reads the rest of the configuration, so do it last
        self.gtest_tests = self._compute_gtest_tests(args)

    @property
    def dry_run(self) -> bool:
        """Whether a dry run is configured."""
        return self.other.dry_run

    @property
    def env(self) -> EnvDict:
        """Custom environment settings used for process exectution."""
        return dict(defaults.PROCESS_ENV)

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
        return Path(os.getcwd())

    @property
    def test_files(self) -> tuple[Path, ...]:
        """List of all test files to use for each stage.

        An explicit list of files from the command line will take precedence.

        Otherwise, the files are computed based on command-line options, etc.

        """
        if self.files and self.last_failed:
            raise RuntimeError("Cannot specify both --files and --last-failed")

        if self.files:
            return self.files

        if self.last_failed:
            if last_failed := self._read_last_failed():
                return last_failed

        files = []

        if self.examples:
            examples = (
                path.relative_to(self.root_dir)
                for path in self.root_dir.joinpath("examples").glob("*.py")
                if str(path.relative_to(self.root_dir)) not in SKIPPED_EXAMPLES
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

        if self.unit:
            unit_tests = (
                path.relative_to(self.root_dir)
                for path in self.root_dir.joinpath("tests/unit").glob(
                    "**/*.py"
                )
            )
            files.extend(sorted(unit_tests))

        return tuple(files)

    @property
    def legate_path(self) -> str:
        """Computed path to the legate driver script"""
        if not hasattr(self, "legate_path_"):

            def compute_legate_path() -> str:
                if self.other.legate_dir is not None:
                    return str(self.other.legate_dir / "bin" / "legate")

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
        if args.features is not None:
            computed = args.features
        else:
            computed = [
                feature
                for feature in FEATURES
                if os.environ.get(f"USE_{feature.upper()}", None) == "1"
            ]

        # if nothing is specified any other way, at least run CPU stage
        if len(computed) == 0:
            computed.append("cpus")

        return tuple(computed)

    def _compute_legate_dir(self, args: Namespace) -> Path | None:
        # self._legate_source below is purely for testing
        if args.legate_dir:
            self._legate_source = "cmd"
            return Path(args.legate_dir)
        elif "LEGATE_DIR" in os.environ:
            self._legate_source = "env"
            return Path(os.environ["LEGATE_DIR"])
        self._legate_source = "install"
        return None

    def _read_last_failed(self) -> tuple[Path, ...]:
        try:
            with open(LAST_FAILED_FILENAME) as f:
                lines = (line for line in f.readlines() if line.strip())
                return tuple(Path(line.strip()) for line in lines)
        except OSError:
            return ()

    def _compute_gtest_tests(self, args: Namespace) -> list[str]:
        if args.gtest_file is None:
            return []

        to_skip = set(args.gtest_skip_list)
        if args.gtest_tests:
            return [test for test in args.gtest_tests if test not in to_skip]

        list_command = [args.gtest_file, "--gtest_list_tests"]
        if args.gtest_filter is not None:
            list_command.append(f"--gtest_filter={args.gtest_filter}")

        try:
            cmd_out = subprocess.check_output(
                list_command, stderr=subprocess.STDOUT
            )
        except subprocess.CalledProcessError as cpe:
            print("Failed to fetch GTest tests")
            if cpe.stdout:
                print(f"stdout:\n{cpe.stdout.decode()}")
            if cpe.stderr:
                print(f"stderr:\n{cpe.stderr.decode()}")
            raise

        result = cmd_out.decode(sys.stdout.encoding).split("\n")

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

            # Skip death tests when running with multiple processes. It looks
            # as if GTest catches the failure and declares the test successful,
            # but for some reason the failure is not actually completely
            # neutralized, and the exit code is non-zero.
            if (
                self.multi_node.ranks_per_node > 1 or self.multi_node.nodes > 1
            ) and (test_group.endswith("DeathTest.")):
                continue

            test_name = test_group + line.split("#")[0].strip()
            if test_name in to_skip:
                continue

            # Assign test to test group
            test_names.append(test_name)

        return test_names
