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

from datetime import timedelta
from pathlib import Path

import pytest

from legate.tester import FeatureType
from legate.tester.config import Config
from legate.tester.stages import test_stage as m
from legate.tester.stages.util import Shard, StageResult, StageSpec
from legate.tester.test_system import ProcessResult, TestSystem as _TestSystem
from legate.util.types import ArgList, EnvDict

from . import FakeSystem


class MockTestStage(m.TestStage):
    kind: FeatureType = "eager"

    name = "mock"

    args = ["-foo", "-bar"]

    def __init__(self, config: Config, system: _TestSystem) -> None:
        self._init(config, system)

    def compute_spec(self, config: Config, system: _TestSystem) -> StageSpec:
        shards = [Shard([(0,)]), Shard([(1,)]), Shard([(2,)])]
        return StageSpec(2, shards)

    def shard_args(self, shard: Shard, config: Config) -> ArgList:
        return []

    def env(self, config: Config, system: _TestSystem) -> EnvDict:
        return {}


class TestTestStage:
    def test_name(self) -> None:
        c = Config([])
        stage = MockTestStage(c, FakeSystem())
        assert stage.name == "mock"

    def test_intro(self) -> None:
        c = Config([])
        stage = MockTestStage(c, FakeSystem())
        assert "Entering stage: mock" in stage.intro

    def test_outro(self) -> None:
        c = Config([])
        stage = MockTestStage(c, FakeSystem())
        stage.result = StageResult(
            [ProcessResult("invoke", Path("test/file"))],
            timedelta(seconds=2.123),
        )
        outro = stage.outro
        assert "Exiting stage: mock" in outro
        assert "Passed 1 of 1 tests (100.0%)" in outro
        assert "2.123" in outro


class TestTestStage_file_args:
    def test_default(self) -> None:
        c = Config([])
        stage = MockTestStage(c, FakeSystem())
        assert stage.file_args(Path("integration/foo"), c) == []
        assert stage.file_args(Path("unit/foo"), c) == []

    def test_v(self) -> None:
        c = Config(["test.py", "-v"])
        stage = MockTestStage(c, FakeSystem())
        assert stage.file_args(Path("integration/foo"), c) == ["-v"]
        assert stage.file_args(Path("unit/foo"), c) == []

    def test_vv(self) -> None:
        c = Config(["test.py", "-vv"])
        stage = MockTestStage(c, FakeSystem())
        assert stage.file_args(Path("integration/foo"), c) == ["-v", "-s"]
        assert stage.file_args(Path("unit/foo"), c) == []


class TestTestStage_cov_args:
    def test_without_cov_bin(self) -> None:
        c = m.Config(["test.py", "--cov-args", "run -a"])
        stage = MockTestStage(c, FakeSystem())
        assert stage.cov_args(c) == []

    def test_with_cov_bin(self) -> None:
        cov_bin = "conda/envs/legate/bin/coverage"
        args = ["--cov-bin", cov_bin]
        c = m.Config(["test.py"] + args)
        expected_result = [
            "--run-mode=python",
            cov_bin,
        ] + c.other.cov_args.split()
        stage = MockTestStage(c, FakeSystem())
        assert stage.cov_args(c) == expected_result

    def test_with_cov_bin_args_and_src_path(self) -> None:
        cov_bin = "conda/envs/legate/bin/coverage"
        cov_args = "run -a"
        cov_src_path = "source_path"
        args = (
            ["--cov-bin", cov_bin]
            + ["--cov-args", cov_args]
            + ["--cov-src-path", cov_src_path]
        )
        c = m.Config(["test.py"] + args)
        expected_result = (
            ["--run-mode=python", cov_bin]
            + cov_args.split()
            + ["--source", cov_src_path]
        )
        stage = MockTestStage(c, FakeSystem())
        assert stage.cov_args(c) == expected_result


class TestTestStage_handle_cpu_pin_args:
    def test_none(self) -> None:
        c = Config(["test.py", "--cpu-pin", "none"])
        stage = MockTestStage(c, FakeSystem())
        shard = Shard([(0, 1), (2, 3)])
        assert stage.handle_cpu_pin_args(c, shard) == []

    def test_strict(self) -> None:
        c = Config(["test.py", "--cpu-pin", "strict"])
        stage = MockTestStage(c, FakeSystem())
        shard = Shard([(0, 1), (2, 3)])
        assert stage.handle_cpu_pin_args(c, shard) == [
            "--cpu-bind",
            str(shard),
        ]

    def test_partial(self) -> None:
        c = Config(["test.py", "--cpu-pin", "partial"])
        stage = MockTestStage(c, FakeSystem())
        shard = Shard([(0, 1), (2, 3)])
        assert stage.handle_cpu_pin_args(c, shard) == [
            "--cpu-bind",
            str(shard),
        ]


class TestTestStage_handle_multi_node_args:
    def test_default(self) -> None:
        c = Config(["test.py"])
        stage = MockTestStage(c, FakeSystem())
        assert stage.handle_multi_node_args(c) == []

    def test_ranks(self) -> None:
        c = Config(["test.py", "--ranks-per-node", "4"])
        stage = MockTestStage(c, FakeSystem())
        assert stage.handle_multi_node_args(c) == [
            "--ranks-per-node",
            "4",
        ]

    def test_nodes(self) -> None:
        c = Config(["test.py", "--nodes", "4"])
        stage = MockTestStage(c, FakeSystem())
        assert stage.handle_multi_node_args(c) == ["--nodes", "4"]

    def test_launcher_none(self) -> None:
        c = Config(["test.py", "--launcher", "none"])
        stage = MockTestStage(c, FakeSystem())
        assert stage.handle_multi_node_args(c) == []

    def test_launcher_mpirun(self) -> None:
        c = Config(["test.py", "--launcher", "mpirun"])
        stage = MockTestStage(c, FakeSystem())
        assert stage.handle_multi_node_args(c) == [
            "--launcher",
            "mpirun",
            "--launcher-extra=--merge-stderr-to-stdout",
        ]

    @pytest.mark.parametrize("launch", ("jsrun", "srun"))
    def test_launcher_others(self, launch: str) -> None:
        c = Config(["test.py", "--launcher", launch])
        stage = MockTestStage(c, FakeSystem())
        assert stage.handle_multi_node_args(c) == ["--launcher", launch]

    def test_launcher_extra(self) -> None:
        c = Config(
            ["test.py", "--launcher-extra", "a/b", "--launcher-extra", "c"]
        )
        stage = MockTestStage(c, FakeSystem())
        assert stage.handle_multi_node_args(c) == [
            "--launcher-extra=a/b",
            "--launcher-extra=c",
        ]

    def test_mpi_output_filename_mpirun(self) -> None:
        c = Config(
            [
                "test.py",
                "--launcher",
                "mpirun",
                "--mpi-output-filename",
                "a/b c/d.out",
            ]
        )
        stage = MockTestStage(c, FakeSystem())
        assert stage.handle_multi_node_args(c) == [
            "--launcher",
            "mpirun",
            "--launcher-extra=--merge-stderr-to-stdout",
            '--launcher-extra="--output-filename"',
            "--launcher-extra='a/b c/d.out'",
        ]

    def test_mpi_output_filename_non_mpirun(self) -> None:
        c = Config(["test.py", "--mpi-output-filename", "a/b c/d.out"])
        stage = MockTestStage(c, FakeSystem())
        assert stage.handle_multi_node_args(c) == []

    def test_combined(self) -> None:
        c = Config(
            [
                "test.py",
                "--nodes",
                "3",
                "--ranks-per-node",
                "2",
                "--launcher",
                "jsrun",
                "--launcher-extra",
                "c",
                "--mpi-output-filename",
                "a/b c/d.out",
            ]
        )
        stage = MockTestStage(c, FakeSystem())
        assert stage.handle_multi_node_args(c) == [
            "--launcher",
            "jsrun",
            "--ranks-per-node",
            "2",
            "--nodes",
            "3",
            "--launcher-extra=c",
        ]
