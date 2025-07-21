# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Consolidate test configuration from command-line and environment."""

from __future__ import annotations

import re
import sys
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import pytest

from legate.tester.config import Config
from legate.tester.defaults import CPU_PIN, SMALL_SYSMEM
from legate.tester.project import Project
from legate.tester.stages._osx import omp as m
from legate.tester.stages.util import UNPIN_ENV, Shard

from .. import FakeSystem

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

unpin_and_test = dict(UNPIN_ENV)

PROJECT = Project()


def get_warn_cpu_pin_context_manager() -> AbstractContextManager[Any]:
    match sys.platform:
        case "darwin":
            # If this is ever not "none", then the CPU.shard_args() function
            # will warn, in case we want this to not be a nullcontext manager.
            assert CPU_PIN == "none"
            return nullcontext()
        case _:
            # If running on other (non macOS systems), the default CPU pinning
            # is something other than "none", meaning that CPU.shard_args()
            # should warn about ignoring CPU pinning arguments
            assert CPU_PIN != "none"
            return pytest.warns(
                UserWarning,
                match=re.escape(
                    "CPU pinning is not supported on macOS, ignoring "
                    "pinning arguments"
                ),
            )


WARN_CPU_PINNING_IF_NOT_MACOS = get_warn_cpu_pin_context_manager()


def test_default() -> None:
    c = Config([], project=PROJECT)
    s = FakeSystem(cpus=12)
    stage = m.OMP(c, s)
    assert stage.kind == "openmp"
    assert stage.args == []
    assert stage.stage_env(c, s) == unpin_and_test
    assert stage.spec.workers > 0

    shard = (1, 2, 3)
    with WARN_CPU_PINNING_IF_NOT_MACOS:
        shard_args = stage.shard_args(Shard([shard]), c)
    assert "--cpus" in shard_args
    assert "--sysmem" in shard_args
    assert "--utility" in shard_args


def test_cpu_pin_strict() -> None:
    c = Config(["test.py", "--cpu-pin", "strict"], project=PROJECT)
    s = FakeSystem(cpus=12)
    stage = m.OMP(c, s)
    assert stage.kind == "openmp"
    assert stage.args == []
    assert stage.stage_env(c, s) == unpin_and_test
    assert stage.spec.workers > 0

    shard = (1, 2, 3)
    with pytest.warns(
        UserWarning,
        match="CPU pinning is not supported on macOS, ignoring "
        "pinning arguments",
    ):
        shard_args = stage.shard_args(Shard([shard]), c)
    assert "--cpus" in shard_args
    assert "--sysmem" in shard_args
    assert "--utility" in shard_args


def test_cpu_pin_strict_zero_computed_workers() -> None:
    num_cores = 12
    c = Config(
        ["test.py", "--cpu-pin", "strict", "--omps", "16"], project=PROJECT
    )
    s = FakeSystem(cpus=num_cores)

    mess = re.escape(
        f"{num_cores} detected core(s) not enough for 1 rank(s) per node, each"
        " reserving 66 core(s) with strict CPU pinning"
    )
    with pytest.raises(RuntimeError, match=mess):
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
    with WARN_CPU_PINNING_IF_NOT_MACOS:
        shard_args = stage.shard_args(Shard([shard]), c)
    assert "--cpus" in shard_args
    assert "--sysmem" in shard_args
    assert "--utility" in shard_args


def test_cpu_pin_none() -> None:
    c = Config(["test.py", "--cpu-pin", "none"], project=PROJECT)
    s = FakeSystem(cpus=12)
    stage = m.OMP(c, s)
    assert stage.kind == "openmp"
    assert stage.args == []
    assert stage.stage_env(c, s) == unpin_and_test
    assert stage.spec.workers > 0

    shard = (1, 2, 3)
    shard_args = stage.shard_args(Shard([shard]), c)
    assert "--cpus" in shard_args
    assert "--sysmem" in shard_args
    assert "--utility" in shard_args


class TestSingleRank:
    @pytest.mark.parametrize("shard", ((2,), (1, 2, 3)))
    def test_shard_args(self, shard: tuple[int, ...]) -> None:
        c = Config([], project=PROJECT)
        s = FakeSystem(cpus=12)
        stage = m.OMP(c, s)
        with WARN_CPU_PINNING_IF_NOT_MACOS:
            result = stage.shard_args(Shard([shard]), c)
        assert result == [
            "--omps",
            f"{c.core.omps}",
            "--ompthreads",
            f"{c.core.ompthreads}",
            "--sysmem",
            str(SMALL_SYSMEM),
            "--cpus",
            "1",
            "--utility",
            f"{c.core.utility}",
        ]

    def test_spec_with_omps_1_threads_1(self) -> None:
        c = Config(
            ["test.py", "--omps", "1", "--ompthreads", "1"], project=PROJECT
        )
        s = FakeSystem(cpus=12)
        stage = m.OMP(c, s)
        assert stage.spec.workers == 6
        assert stage.spec.shards == [
            Shard([(0,)]),
            Shard([(1,)]),
            Shard([(2,)]),
            Shard([(3,)]),
            Shard([(4,)]),
            Shard([(5,)]),
        ]

    def test_spec_with_omps_1_threads_2(self) -> None:
        c = Config(
            ["test.py", "--omps", "1", "--ompthreads", "2"], project=PROJECT
        )
        s = FakeSystem(cpus=12)
        stage = m.OMP(c, s)
        assert stage.spec.workers == 4
        assert stage.spec.shards == [
            Shard([(0,)]),
            Shard([(1,)]),
            Shard([(2,)]),
            Shard([(3,)]),
        ]

    def test_spec_with_omps_2_threads_1(self) -> None:
        c = Config(
            ["test.py", "--omps", "2", "--ompthreads", "1"], project=PROJECT
        )
        s = FakeSystem(cpus=12)
        stage = m.OMP(c, s)
        assert stage.spec.workers == 4
        assert stage.spec.shards == [
            Shard([(0,)]),
            Shard([(1,)]),
            Shard([(2,)]),
            Shard([(3,)]),
        ]

    def test_spec_with_omps_2_threads_2(self) -> None:
        c = Config(
            ["test.py", "--omps", "2", "--ompthreads", "2"], project=PROJECT
        )
        s = FakeSystem(cpus=12)
        stage = m.OMP(c, s)
        assert stage.spec.workers == 2
        assert stage.spec.shards == [Shard([(0,)]), Shard([(1,)])]

    def test_spec_with_utility(self) -> None:
        c = Config(
            ["test.py", "--omps", "2", "--ompthreads", "2", "--utility", "3"],
            project=PROJECT,
        )
        s = FakeSystem(cpus=12)
        stage = m.OMP(c, s)
        assert stage.spec.workers == 1
        assert stage.spec.shards == [Shard([(0,)])]

    def test_spec_with_requested_workers(self) -> None:
        c = Config(
            ["test.py", "--omps", "1", "--ompthreads", "1", "-j", "2"],
            project=PROJECT,
        )
        s = FakeSystem(cpus=12)
        stage = m.OMP(c, s)
        assert stage.spec.workers == 2
        assert stage.spec.shards == [Shard([(0,)]), Shard([(1,)])]

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
        num_cpus = 4
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
        s = FakeSystem(cpus=num_cpus)

        mess = re.escape(
            rf"{num_cpus} detected core(s) not enough for 1 rank(s) per node, "
            rf"each reserving {threads + 1} core(s), running anyway."
        )
        with pytest.warns(UserWarning, match=mess):
            stage = m.OMP(c, s)

        assert stage.spec.workers == 1
        assert stage.spec.shards == [Shard([(0, 1, 2, 3)])]


class TestMultiRank:
    @pytest.mark.parametrize("shard", ((2,), (1, 2, 3)))
    def test_shard_args(self, shard: tuple[int, ...]) -> None:
        c = Config([], project=PROJECT)
        s = FakeSystem(cpus=12)
        stage = m.OMP(c, s)
        with WARN_CPU_PINNING_IF_NOT_MACOS:
            result = stage.shard_args(Shard([shard]), c)
        assert result == [
            "--omps",
            f"{c.core.omps}",
            "--ompthreads",
            f"{c.core.ompthreads}",
            "--sysmem",
            str(SMALL_SYSMEM),
            "--cpus",
            "1",
            "--utility",
            f"{c.core.utility}",
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
            Shard([(0,)]),
            Shard([(1,)]),
            Shard([(2,)]),
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
        assert stage.spec.shards == [Shard([(0,)]), Shard([(1,)])]

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
        assert stage.spec.shards == [Shard([(0,)]), Shard([(1,)])]

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
        assert stage.spec.shards == [Shard([(0,)])]

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
        assert stage.spec.shards == [Shard([(0,)])]

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
        assert stage.spec.shards == [Shard([(0,)]), Shard([(1,)])]

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
        ranks_per_node = 2
        num_cpus = 4
        args = [
            "test.py",
            "--omps",
            "1",
            "--ompthreads",
            str(threads),
            "--ranks-per-node",
            str(ranks_per_node),
            "--cpu-pin",
            "none",
            # any launcher will do
            "--launcher",
            "srun",
        ]
        c = Config(args, project=PROJECT)
        s = FakeSystem(cpus=num_cpus)

        mess = re.escape(
            rf"{num_cpus} detected core(s) not enough for {ranks_per_node} "
            rf"rank(s) per node, each reserving {threads + 1} core(s), "
            r"running anyway."
        )
        with pytest.warns(UserWarning, match=mess):
            stage = m.OMP(c, s)

        assert stage.spec.workers == 1
        assert stage.spec.shards == [Shard([(0, 1, 2, 3)])]
