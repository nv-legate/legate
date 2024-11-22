# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

import pytest
from rich.console import Console

from legate.tester import FeatureType, defaults
from legate.tester.config import Config
from legate.tester.project import Project
from legate.tester.stages import test_stage as m
from legate.tester.stages.util import (
    MANUAL_CONFIG_ENV,
    Shard,
    StageResult,
    StageSpec,
)
from legate.tester.test_system import ProcessResult, TestSystem as _TestSystem
from legate.util.types import ArgList, EnvDict

from . import FakeSystem

CONSOLE = Console(color_system=None, soft_wrap=True)


class MockTestStage(m.TestStage):
    name = "mock"

    args = ["-foo", "-bar"]

    def __init__(
        self, config: Config, system: _TestSystem, kind: FeatureType = "eager"
    ) -> None:
        self.kind = kind
        self._init(config, system)

    def compute_spec(self, config: Config, system: _TestSystem) -> StageSpec:
        shards = [Shard([(0,)]), Shard([(1,)]), Shard([(2,)])]
        return StageSpec(2, shards)

    def shard_args(self, shard: Shard, config: Config) -> ArgList:
        return []

    def stage_env(self, config: Config, system: _TestSystem) -> EnvDict:
        return {"stage": "env"}


class TestTestStage:
    def test_name(self) -> None:
        c = Config([])
        stage = MockTestStage(c, FakeSystem())
        assert stage.name == "mock"

    def test_intro(self) -> None:
        c = Config([])
        stage = MockTestStage(c, FakeSystem())

        with CONSOLE.capture() as capture:
            CONSOLE.print(stage.intro)
        assert "Entering stage: mock" in capture.get()

    def test_outro(self) -> None:
        c = Config([])
        stage = MockTestStage(c, FakeSystem())
        stage.result = StageResult(
            [ProcessResult("invoke", "test/file")],
            timedelta(seconds=2.12),
        )
        with CONSOLE.capture() as capture:
            CONSOLE.print(stage.outro)
        outro = capture.get()
        assert "Exiting stage: mock" in outro
        assert "Passed 1 of 1 tests (100.0%)" in outro
        assert "2.12" in outro

    def test_env(self) -> None:
        c = Config([])
        s = FakeSystem()
        stage = MockTestStage(c, s)

        env = stage.env(c, FakeSystem())

        expected = dict(defaults.PROCESS_ENV)
        expected.update(MANUAL_CONFIG_ENV)
        expected.update(stage.stage_env(c, s))
        # no project.stage_env

        assert env == expected
        assert "LEGATE_CONFIG" not in env

    def test_env_with_LEGATE_CONFIG(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LEGATE_CONFIG", "foo")

        c = Config([])
        s = FakeSystem()
        stage = MockTestStage(c, s)

        env = stage.env(c, FakeSystem())

        expected = dict(defaults.PROCESS_ENV)
        expected.update(MANUAL_CONFIG_ENV)
        expected.update(stage.stage_env(c, s))
        # no project.stage_env
        expected["LEGATE_CONFIG"] = "foo"

        assert env == expected

    @pytest.mark.parametrize("feature", defaults.FEATURES)
    def test_env_with_custom_project(self, feature: FeatureType) -> None:
        class CustomProj(Project):
            def stage_env(self, feature: FeatureType) -> EnvDict:
                return {"feature": feature}

        c = Config([], CustomProj())
        s = FakeSystem()
        stage = MockTestStage(c, s, feature)

        env = stage.env(c, FakeSystem())

        expected = dict(defaults.PROCESS_ENV)
        expected.update(MANUAL_CONFIG_ENV)
        expected.update(stage.stage_env(c, s))
        expected.update(c.project.stage_env(feature))

        assert env["feature"] == feature
        assert env == expected


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

    @pytest.mark.parametrize("launch", ("jsrun", "srun"))
    def test_ranks(self, launch: str) -> None:
        c = Config(["test.py", "--ranks-per-node", "4", "--launcher", launch])
        stage = MockTestStage(c, FakeSystem())
        assert stage.handle_multi_node_args(c) == [
            "--launcher",
            launch,
            "--ranks-per-node",
            "4",
        ]

    @pytest.mark.parametrize("launch", ("jsrun", "srun"))
    def test_nodes(self, launch: str) -> None:
        c = Config(["test.py", "--nodes", "4", "--launcher", launch])
        stage = MockTestStage(c, FakeSystem())
        assert stage.handle_multi_node_args(c) == [
            "--launcher",
            launch,
            "--nodes",
            "4",
        ]

    def test_launcher_none(self) -> None:
        c = Config(["test.py", "--launcher", "none"])
        stage = MockTestStage(c, FakeSystem())
        assert stage.handle_multi_node_args(c) == []

    @pytest.mark.parametrize("launch", ("jsrun", "srun", "mpirun"))
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
