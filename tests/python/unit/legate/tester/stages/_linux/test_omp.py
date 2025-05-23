# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Consolidate test configuration from command-line and environment."""

from __future__ import annotations

import pytest

from legate.tester.config import Config
from legate.tester.defaults import SMALL_SYSMEM
from legate.tester.project import Project
from legate.tester.stages._linux import omp as m
from legate.tester.stages.util import UNPIN_ENV, Shard

from .. import FakeSystem

unpin_and_test = dict(UNPIN_ENV)

PROJECT = Project()


def test_default() -> None:
    c = Config([], project=PROJECT)
    s = FakeSystem(cpus=12)
    stage = m.OMP(c, s)
    assert stage.kind == "openmp"
    assert stage.args == []
    assert stage.stage_env(c, s) == unpin_and_test
    assert stage.spec.workers > 0

    shard = (1, 2, 3)
    assert "--cpu-bind" in stage.shard_args(Shard([shard]), c)


def test_cpu_pin_strict() -> None:
    c = Config(["test.py", "--cpu-pin", "strict"], project=PROJECT)
    s = FakeSystem(cpus=12)
    stage = m.OMP(c, s)
    assert stage.kind == "openmp"
    assert stage.args == []
    assert stage.stage_env(c, s) == {}
    assert stage.spec.workers > 0

    shard = (1, 2, 3)
    assert "--cpu-bind" in stage.shard_args(Shard([shard]), c)


def test_cpu_pin_strict_zero_computed_workers() -> None:
    c = Config(
        ["test.py", "--cpu-pin", "strict", "--omps", "16"], project=PROJECT
    )
    s = FakeSystem(cpus=12)
    with pytest.raises(RuntimeError, match="not enough"):
        m.OMP(c, s)


@pytest.mark.filterwarnings(
    r"ignore:\d+ detected core\(s\) not enough for.*running anyway"
)
def test_cpu_pin_nonstrict_zero_computed_workers() -> None:
    c = Config(["test.py", "--omps", "16"], project=PROJECT)
    s = FakeSystem(cpus=12)
    stage = m.OMP(c, s)
    assert stage.kind == "openmp"
    assert stage.args == []
    assert stage.stage_env(c, s) == unpin_and_test
    assert stage.spec.workers == 1

    shard = tuple(range(12))
    assert "--cpu-bind" in stage.shard_args(Shard([shard]), c)


def test_cpu_pin_none() -> None:
    c = Config(["test.py", "--cpu-pin", "none"], project=PROJECT)
    s = FakeSystem(cpus=12)
    stage = m.OMP(c, s)
    assert stage.kind == "openmp"
    assert stage.args == []
    assert stage.stage_env(c, s) == unpin_and_test
    assert stage.spec.workers > 0

    shard = (1, 2, 3)
    assert "--cpu-bind" not in stage.shard_args(Shard([shard]), c)


class TestSingleRank:
    @pytest.mark.parametrize(
        ("shard", "expected"), [[(2,), "2"], [(1, 2, 3), "1,2,3"]]
    )
    def test_shard_args(self, shard: tuple[int, ...], expected: str) -> None:
        c = Config([], project=PROJECT)
        s = FakeSystem(cpus=12)
        stage = m.OMP(c, s)
        result = stage.shard_args(Shard([shard]), c)
        assert result == [
            "--omps",
            f"{c.core.omps}",
            "--ompthreads",
            f"{c.core.ompthreads}",
            "--numamem",
            f"{c.memory.numamem}",
            "--sysmem",
            str(SMALL_SYSMEM),
            "--cpus",
            "1",
            "--utility",
            f"{c.core.utility}",
            "--cpu-bind",
            expected,
        ]

    def test_spec_with_omps_1_threads_1(self) -> None:
        c = Config(
            ["test.py", "--omps", "1", "--ompthreads", "1"], project=PROJECT
        )
        s = FakeSystem(cpus=12)
        stage = m.OMP(c, s)
        assert stage.spec.workers == 6
        assert stage.spec.shards == [
            Shard([(0, 1)]),
            Shard([(2, 3)]),
            Shard([(4, 5)]),
            Shard([(6, 7)]),
            Shard([(8, 9)]),
            Shard([(10, 11)]),
        ]

    def test_spec_with_omps_1_threads_2(self) -> None:
        c = Config(
            ["test.py", "--omps", "1", "--ompthreads", "2"], project=PROJECT
        )
        s = FakeSystem(cpus=12)
        stage = m.OMP(c, s)
        assert stage.spec.workers == 4
        assert stage.spec.shards == [
            Shard([(0, 1, 2)]),
            Shard([(3, 4, 5)]),
            Shard([(6, 7, 8)]),
            Shard([(9, 10, 11)]),
        ]

    def test_spec_with_omps_2_threads_1(self) -> None:
        c = Config(
            ["test.py", "--omps", "2", "--ompthreads", "1"], project=PROJECT
        )
        s = FakeSystem(cpus=12)
        stage = m.OMP(c, s)
        assert stage.spec.workers == 4
        assert stage.spec.shards == [
            Shard([(0, 1, 2)]),
            Shard([(3, 4, 5)]),
            Shard([(6, 7, 8)]),
            Shard([(9, 10, 11)]),
        ]

    def test_spec_with_omps_2_threads_2(self) -> None:
        c = Config(
            ["test.py", "--omps", "2", "--ompthreads", "2"], project=PROJECT
        )
        s = FakeSystem(cpus=12)
        stage = m.OMP(c, s)
        assert stage.spec.workers == 2
        assert stage.spec.shards == [
            Shard([(0, 1, 2, 3, 4)]),
            Shard([(5, 6, 7, 8, 9)]),
        ]

    def test_spec_with_utility(self) -> None:
        c = Config(
            ["test.py", "--omps", "2", "--ompthreads", "2", "--utility", "3"],
            project=PROJECT,
        )
        s = FakeSystem(cpus=12)
        stage = m.OMP(c, s)
        assert stage.spec.workers == 1
        assert stage.spec.shards == [Shard([(0, 1, 2, 3, 4, 5, 6)])]

    def test_spec_with_requested_workers(self) -> None:
        c = Config(
            ["test.py", "--omps", "1", "--ompthreads", "1", "-j", "2"],
            project=PROJECT,
        )
        s = FakeSystem(cpus=12)
        stage = m.OMP(c, s)
        assert stage.spec.workers == 2
        assert stage.spec.shards == [Shard([(0, 1)]), Shard([(2, 3)])]

    def test_spec_with_requested_workers_zero(self) -> None:
        s = FakeSystem(cpus=12)
        c = Config(["test.py", "-j", "0"], project=PROJECT)
        assert c.execution.workers == 0
        with pytest.raises(RuntimeError):
            m.OMP(c, s)

    def test_spec_with_requested_workers_bad(self) -> None:
        s = FakeSystem(cpus=12)
        c = Config(["test.py", "-j", f"{len(s.cpus) + 1}"], project=PROJECT)
        requested_workers = c.execution.workers
        assert requested_workers is not None
        assert requested_workers > len(s.cpus)
        with pytest.raises(RuntimeError):
            m.OMP(c, s)

    def test_spec_with_verbose(self) -> None:
        args = ["test.py", "--cpus", "2"]
        c = Config(args, project=PROJECT)
        cv = Config([*args, "--verbose"], project=PROJECT)
        s = FakeSystem(cpus=12)

        spec, vspec = m.OMP(c, s).spec, m.OMP(cv, s).spec
        assert vspec == spec

    @pytest.mark.parametrize("threads", (4, 5, 10, 20))
    def test_oversubscription_with_pin(self, threads: int) -> None:
        args = [
            "test.py",
            "--omps",
            "1",
            "--ompthreads",
            str(threads),
            "--cpu-pin",
            "strict",
        ]
        c = Config(args, project=PROJECT)
        s = FakeSystem(cpus=4)

        with pytest.raises(RuntimeError):
            m.OMP(c, s)

    @pytest.mark.parametrize("threads", (4, 5, 10, 20))
    def test_oversubscription_no_pin(self, threads: int) -> None:
        args = [
            "test.py",
            "--omps",
            "1",
            "--ompthreads",
            str(threads),
            "--cpu-pin",
            "none",
        ]
        c = Config(args, project=PROJECT)
        s = FakeSystem(cpus=4)

        with pytest.warns():
            stage = m.OMP(c, s)

        assert stage.spec.workers == 1
        assert stage.spec.shards == [Shard([(0, 1, 2, 3)])]


class TestMultiRank:
    @pytest.mark.parametrize(
        ("shard", "expected"), [[(2,), "2"], [(1, 2, 3), "1,2,3"]]
    )
    def test_shard_args(self, shard: tuple[int, ...], expected: str) -> None:
        c = Config([], project=PROJECT)
        s = FakeSystem(cpus=12)
        stage = m.OMP(c, s)
        result = stage.shard_args(Shard([shard]), c)
        assert result == [
            "--omps",
            f"{c.core.omps}",
            "--ompthreads",
            f"{c.core.ompthreads}",
            "--numamem",
            f"{c.memory.numamem}",
            "--sysmem",
            str(SMALL_SYSMEM),
            "--cpus",
            "1",
            "--utility",
            f"{c.core.utility}",
            "--cpu-bind",
            expected,
        ]

    def test_spec_with_omps_1_threads_1(self) -> None:
        c = Config(
            [
                "test.py",
                "--omps",
                "1",
                "--ompthreads",
                "1",
                "--ranks-per-node",
                "2",
                # any launcher will do
                "--launcher",
                "srun",
            ],
            project=PROJECT,
        )
        s = FakeSystem(cpus=12)
        stage = m.OMP(c, s)
        assert stage.spec.workers == 3
        assert stage.spec.shards == [
            Shard([(0, 1), (2, 3)]),
            Shard([(4, 5), (6, 7)]),
            Shard([(8, 9), (10, 11)]),
        ]

    def test_spec_with_omps_1_threads_2(self) -> None:
        c = Config(
            [
                "test.py",
                "--omps",
                "1",
                "--ompthreads",
                "2",
                "--ranks-per-node",
                "2",
                # any launcher will do
                "--launcher",
                "srun",
            ],
            project=PROJECT,
        )
        s = FakeSystem(cpus=12)
        stage = m.OMP(c, s)
        assert stage.spec.workers == 2
        assert stage.spec.shards == [
            Shard([(0, 1, 2), (3, 4, 5)]),
            Shard([(6, 7, 8), (9, 10, 11)]),
        ]

    def test_spec_with_omps_2_threads_1(self) -> None:
        c = Config(
            [
                "test.py",
                "--omps",
                "2",
                "--ompthreads",
                "1",
                "--ranks-per-node",
                "2",
                # any launcher will do
                "--launcher",
                "srun",
            ],
            project=PROJECT,
        )
        s = FakeSystem(cpus=12)
        stage = m.OMP(c, s)
        assert stage.spec.workers == 2
        assert stage.spec.shards == [
            Shard([(0, 1, 2), (3, 4, 5)]),
            Shard([(6, 7, 8), (9, 10, 11)]),
        ]

    def test_spec_with_omps_2_threads_2(self) -> None:
        c = Config(
            [
                "test.py",
                "--omps",
                "2",
                "--ompthreads",
                "2",
                "--ranks-per-node",
                "2",
                # any launcher will do
                "--launcher",
                "srun",
            ],
            project=PROJECT,
        )
        s = FakeSystem(cpus=12)
        stage = m.OMP(c, s)
        assert stage.spec.workers == 1
        assert stage.spec.shards == [Shard([(0, 1, 2, 3, 4), (5, 6, 7, 8, 9)])]

    def test_spec_with_utility(self) -> None:
        c = Config(
            [
                "test.py",
                "--omps",
                "2",
                "--ompthreads",
                "2",
                "--utility",
                "3",
                "--ranks-per-node",
                "2",
                # any launcher will do
                "--launcher",
                "srun",
            ],
            project=PROJECT,
        )
        s = FakeSystem(cpus=16)
        stage = m.OMP(c, s)
        assert stage.spec.workers == 1
        assert stage.spec.shards == [
            Shard([(0, 1, 2, 3, 4, 5, 6), (7, 8, 9, 10, 11, 12, 13)])
        ]

    def test_spec_with_requested_workers(self) -> None:
        c = Config(
            [
                "test.py",
                "--omps",
                "1",
                "--ompthreads",
                "1",
                "-j",
                "2",
                "--ranks-per-node",
                "2",
                # any launcher will do
                "--launcher",
                "srun",
            ],
            project=PROJECT,
        )
        s = FakeSystem(cpus=12)
        stage = m.OMP(c, s)
        assert stage.spec.workers == 2
        assert stage.spec.shards == [
            Shard([(0, 1), (2, 3)]),
            Shard([(4, 5), (6, 7)]),
        ]

    @pytest.mark.parametrize("threads", (2, 3, 10, 20))
    def test_oversubscription_with_pin(self, threads: int) -> None:
        args = [
            "test.py",
            "--omps",
            "1",
            "--ompthreads",
            str(threads),
            "--ranks-per-node",
            "2",
            "--cpu-pin",
            "strict",
            # any launcher will do
            "--launcher",
            "srun",
        ]
        c = Config(args, project=PROJECT)
        s = FakeSystem(cpus=4)

        with pytest.raises(RuntimeError):
            m.OMP(c, s)

    @pytest.mark.parametrize("threads", (2, 3, 10, 20))
    def test_oversubscription_no_pin(self, threads: int) -> None:
        args = [
            "test.py",
            "--omps",
            "1",
            "--ompthreads",
            str(threads),
            "--ranks-per-node",
            "2",
            "--cpu-pin",
            "none",
            # any launcher will do
            "--launcher",
            "srun",
        ]
        c = Config(args, project=PROJECT)
        s = FakeSystem(cpus=4)

        with pytest.warns():
            stage = m.OMP(c, s)

        assert stage.spec.workers == 1
        assert stage.spec.shards == [Shard([(0, 1, 2, 3)])]
