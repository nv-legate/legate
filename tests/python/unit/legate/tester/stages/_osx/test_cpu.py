# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import re
import sys
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import pytest

from legate.tester.config import Config
from legate.tester.defaults import CPU_PIN
from legate.tester.project import Project
from legate.tester.stages._osx import cpu as m
from legate.tester.stages.util import UNPIN_ENV, Shard

from .. import FakeSystem

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

PROJECT = Project()

unpin_and_test = dict(UNPIN_ENV)


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
    stage = m.CPU(c, s)
    assert stage.kind == "cpus"
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
    stage = m.CPU(c, s)
    assert stage.kind == "cpus"
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
    c = Config(
        ["test.py", "--cpu-pin", "strict", "--cpus", "16"], project=PROJECT
    )
    s = FakeSystem(cpus=12)
    mess = re.escape(
        "While CPU pinning is not supported in macOS, this configuration is "
        "nevertheless unsatisfiable. If you would like legate to launch it "
        "anyway, run with '--cpu-pin none'"
    )
    with pytest.raises(RuntimeError, match=mess):
        m.CPU(c, s)


@pytest.mark.filterwarnings(
    r"ignore:\d+ detected core\(s\) not enough for.*running anyway"
)
def test_cpu_pin_nonstrict_zero_computed_workers() -> None:
    c = Config(["test.py", "--cpus", "16"], project=PROJECT)
    s = FakeSystem(cpus=12)
    stage = m.CPU(c, s)
    assert stage.kind == "cpus"
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
    stage = m.CPU(c, s)
    assert stage.kind == "cpus"
    assert stage.args == []
    assert stage.stage_env(c, s) == unpin_and_test
    assert stage.spec.workers > 0

    shard = (1, 2, 3)
    assert "--cpu-bind" not in stage.shard_args(Shard([shard]), c)


class TestSingleRank:
    @pytest.mark.parametrize("shard", ((2,), (1, 2, 3)))
    def test_shard_args(self, shard: tuple[int, ...]) -> None:
        c = Config(["test.py", "--sysmem", "2000"], project=PROJECT)
        s = FakeSystem()
        stage = m.CPU(c, s)
        with WARN_CPU_PINNING_IF_NOT_MACOS:
            result = stage.shard_args(Shard([shard]), c)
        assert result == [
            "--cpus",
            f"{c.core.cpus}",
            "--sysmem",
            str(c.memory.sysmem),
            "--utility",
            f"{c.core.utility}",
            # Note: missing --cpu-bind arguments, not support on macOS
        ]

    def test_spec_with_cpus_1(self) -> None:
        c = Config(["test.py", "--cpus", "1"], project=PROJECT)
        s = FakeSystem()
        stage = m.CPU(c, s)
        assert stage.spec.workers == 3
        assert stage.spec.shards == [
            Shard([(0,)]),
            Shard([(1,)]),
            Shard([(2,)]),
        ]

    def test_spec_with_cpus_2(self) -> None:
        c = Config(["test.py", "--cpus", "2"], project=PROJECT)
        s = FakeSystem()
        stage = m.CPU(c, s)
        assert stage.spec.workers == 2
        assert stage.spec.shards == [Shard([(0,)]), Shard([(1,)])]

    def test_spec_with_utility(self) -> None:
        c = Config(
            ["test.py", "--cpus", "1", "--utility", "2"], project=PROJECT
        )
        s = FakeSystem()
        stage = m.CPU(c, s)
        assert stage.spec.workers == 2
        assert stage.spec.shards == [Shard([(0,)]), Shard([(1,)])]

    def test_spec_with_requested_workers(self) -> None:
        c = Config(["test.py", "--cpus", "1", "-j", "2"], project=PROJECT)
        s = FakeSystem()
        stage = m.CPU(c, s)
        assert stage.spec.workers == 2
        assert stage.spec.shards == [Shard([(0,)]), Shard([(1,)])]

    def test_spec_with_requested_workers_zero(self) -> None:
        s = FakeSystem()
        c = Config(["test.py", "-j", "0"], project=PROJECT)
        assert c.execution.workers == 0
        with pytest.raises(RuntimeError):
            m.CPU(c, s)

    def test_spec_with_requested_workers_bad(self) -> None:
        s = FakeSystem()
        c = Config(["test.py", "-j", f"{len(s.cpus) + 1}"], project=PROJECT)
        requested_workers = c.execution.workers
        assert requested_workers is not None
        assert requested_workers > len(s.cpus)
        with pytest.raises(RuntimeError):
            m.CPU(c, s)

    def test_spec_with_verbose(self) -> None:
        args = ["test.py", "--cpus", "2"]
        c = Config(args, project=PROJECT)
        cv = Config([*args, "--verbose"], project=PROJECT)
        s = FakeSystem()

        spec = m.CPU(c, s).spec
        vspec = m.CPU(cv, s).spec
        assert vspec == spec

    @pytest.mark.parametrize("cpus", (4, 5, 10, 20))
    def test_oversubscription_with_pin(self, cpus: int) -> None:
        args = ["test.py", "--cpus", str(cpus), "--cpu-pin", "strict"]
        c = Config(args, project=PROJECT)
        s = FakeSystem(cpus=4)

        with pytest.raises(RuntimeError):
            m.CPU(c, s)

    @pytest.mark.parametrize("cpus", (4, 5, 10, 20))
    def test_oversubscription_no_pin(self, cpus: int) -> None:
        num_cpus = 4
        args = ["test.py", "--cpus", str(cpus), "--cpu-pin", "none"]
        c = Config(args, project=PROJECT)
        s = FakeSystem(cpus=num_cpus)

        mess = re.escape(
            rf"{num_cpus} detected core(s) not enough for 1 rank(s) per node, "
            rf"each reserving {cpus + 1} core(s), running anyway."
        )
        with pytest.warns(UserWarning, match=mess):
            stage = m.CPU(c, s)

        assert stage.spec.workers == 1
        assert stage.spec.shards == [Shard([(0, 1, 2, 3)])]


class TestMultiRank:
    def test_shard_args(self) -> None:
        c = Config(
            [
                "test.py",
                "--cpus",
                "2",
                "--ranks-per-node",
                "2",
                "--sysmem",
                "2000",
                # any launcher will do
                "--launcher",
                "srun",
            ],
            project=PROJECT,
        )
        s = FakeSystem(cpus=12)
        stage = m.CPU(c, s)
        with WARN_CPU_PINNING_IF_NOT_MACOS:
            result = stage.shard_args(Shard([(0, 1), (2, 3)]), c)
        assert result == [
            "--cpus",
            f"{c.core.cpus}",
            "--sysmem",
            str(c.memory.sysmem),
            "--utility",
            f"{c.core.utility}",
            "--launcher",
            "srun",
            "--ranks-per-node",
            "2",
        ]

    def test_spec_with_cpus_1(self) -> None:
        c = Config(
            [
                "test.py",
                "--cpus",
                "1",
                "--ranks-per-node",
                "2",
                "--launcher",
                "srun",
            ],
            project=PROJECT,
        )
        s = FakeSystem(cpus=12)
        stage = m.CPU(c, s)
        assert stage.spec.workers == 3
        assert stage.spec.shards == [
            Shard([(0,)]),
            Shard([(1,)]),
            Shard([(2,)]),
        ]

    def test_spec_with_cpus_2(self) -> None:
        c = Config(
            [
                "test.py",
                "--cpus",
                "2",
                "--ranks-per-node",
                "2",
                "--launcher",
                "srun",
            ],
            project=PROJECT,
        )
        s = FakeSystem(cpus=12)
        stage = m.CPU(c, s)
        assert stage.spec.workers == 2
        assert stage.spec.shards == [Shard([(0,)]), Shard([(1,)])]

    def test_spec_with_utility_2(self) -> None:
        c = Config(
            [
                "test.py",
                "--cpus",
                "1",
                "--utility",
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
        stage = m.CPU(c, s)
        assert stage.spec.workers == 2
        assert stage.spec.shards == [Shard([(0,)]), Shard([(1,)])]

    def test_spec_with_requested_workers(self) -> None:
        c = Config(
            [
                "test.py",
                "--cpus",
                "1",
                "-j",
                "1",
                "--ranks-per-node",
                "2",
                "--launcher",
                "srun",
            ],
            project=PROJECT,
        )
        s = FakeSystem(cpus=12)
        stage = m.CPU(c, s)
        assert stage.spec.workers == 1
        assert stage.spec.shards == [Shard([(0,)])]

    def test_spec_with_requested_workers_zero(self) -> None:
        s = FakeSystem(cpus=12)
        c = Config(
            [
                "test.py",
                "-j",
                "0",
                "--ranks-per-node",
                "2",
                "--launcher",
                "srun",
            ],
            project=PROJECT,
        )
        assert c.execution.workers == 0
        with pytest.raises(RuntimeError):
            m.CPU(c, s)

    def test_spec_with_requested_workers_bad(self) -> None:
        s = FakeSystem(cpus=12)
        c = Config(
            [
                "test.py",
                "-j",
                f"{len(s.cpus) + 1}",
                "--ranks-per-node",
                "2",
                "--launcher",
                "srun",
            ],
            project=PROJECT,
        )
        requested_workers = c.execution.workers
        assert requested_workers is not None
        assert requested_workers > len(s.cpus)
        with pytest.raises(RuntimeError):
            m.CPU(c, s)

    @pytest.mark.parametrize("cpus", (2, 3, 10, 20))
    def test_oversubscription_with_pin(self, cpus: int) -> None:
        args = [
            "test.py",
            "--cpus",
            str(cpus),
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
            m.CPU(c, s)

    @pytest.mark.parametrize("cpus", (2, 3, 10, 20))
    def test_oversubscription_no_pin(self, cpus: int) -> None:
        ranks_per_node = 2
        num_cpus = 4
        args = [
            "test.py",
            "--cpus",
            str(cpus),
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
            rf"rank(s) per node, each reserving {cpus + 1} core(s), running "
            r"anyway."
        )
        with pytest.warns(UserWarning, match=mess):
            stage = m.CPU(c, s)

        assert stage.spec.workers == 1
        assert stage.spec.shards == [Shard([(0, 1, 2, 3)])]
