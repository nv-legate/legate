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

import pytest

from legate.tester.config import Config
from legate.tester.defaults import SMALL_SYSMEM
from legate.tester.stages._linux import eager as m
from legate.tester.stages.util import Shard

from .. import FakeSystem


def test_default() -> None:
    c = Config([])
    s = FakeSystem()
    stage = m.Eager(c, s)
    assert stage.kind == "eager"
    assert stage.args == []
    assert stage.env(c, s) == {
        "LEGATE_AUTO_CONFIG": "0",
        "CUPYNUMERIC_FORCE_THUNK": "eager",
        "CUPYNUMERIC_MIN_CPU_CHUNK": "2000000000",
        "CUPYNUMERIC_MIN_OMP_CHUNK": "2000000000",
        "CUPYNUMERIC_MIN_GPU_CHUNK": "2000000000",
    }
    assert stage.spec.workers > 0


@pytest.mark.parametrize("shard,expected", [[(2,), "2"], [(1, 2, 3), "1,2,3"]])
def test_single_rank_shard_args(shard: tuple[int, ...], expected: str) -> None:
    c = Config([])
    s = FakeSystem()
    stage = m.Eager(c, s)
    result = stage.shard_args(Shard([shard]), c)
    assert result == [
        "--cpus",
        "1",
        "--cpu-bind",
        expected,
        "--sysmem",
        str(SMALL_SYSMEM),
        "--utility",
        f"{c.core.utility}",
    ]


def test_single_rank_spec() -> None:
    c = Config([])
    s = FakeSystem()
    stage = m.Eager(c, s)
    assert stage.spec.workers == len(s.cpus)
    #  [cpu.ids for cpu in system.cpus]
    expected = [Shard([(i,)]) for i in range(stage.spec.workers)]
    assert stage.spec.shards == expected


def test_single_rank_spec_with_requested_workers_zero() -> None:
    s = FakeSystem()
    c = Config(["test.py", "-j", "0"])
    assert c.execution.workers == 0
    with pytest.raises(RuntimeError):
        m.Eager(c, s)


def test_single_rank_spec_with_requested_workers_bad() -> None:
    s = FakeSystem()
    c = Config(["test.py", "-j", f"{len(s.cpus) + 1}"])
    requested_workers = c.execution.workers
    assert requested_workers is not None
    assert requested_workers > len(s.cpus)
    with pytest.raises(RuntimeError):
        m.Eager(c, s)


def test_single_rank_spec_with_verbose() -> None:
    c = Config(["test.py"])
    cv = Config(["test.py", "--verbose"])
    s = FakeSystem()

    spec, vspec = m.Eager(c, s).spec, m.Eager(cv, s).spec
    assert vspec == spec
