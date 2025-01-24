# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
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

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import call

import legate.jupyter.config as m
from legate.driver import defaults
from legate.driver.config import Core, Memory, MultiNode
from legate.util.types import DataclassMixin

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


class TestKernel:
    def test_fields(self) -> None:
        assert set(m.Kernel.__dataclass_fields__) == {
            "user",
            "prefix",
            "spec_name",
            "display_name",
        }

    def test_mixin(self) -> None:
        assert issubclass(m.Kernel, DataclassMixin)


class TestConfig:
    def test_default_init(self) -> None:
        # Note this test does not clear the environment. Default values from
        # the defaults module can depend on the environment, but what matters
        # is that the generated config matches those values, whatever they are.

        c = m.Config(["legate-jupyter"])

        assert c.multi_node == m.MultiNode(
            nodes=defaults.LEGATE_NODES,
            ranks_per_node=defaults.LEGATE_RANKS_PER_NODE,
            launcher="none",
            launcher_extra=[],
        )
        assert c.core == m.Core(
            cpus=None, gpus=None, omps=None, ompthreads=None, utility=None
        )
        assert c.memory == m.Memory(
            sysmem=None, numamem=None, fbmem=None, zcmem=None, regmem=None
        )

        # These are all "turned off"

        assert c.binding == m.Binding(
            cpu_bind=None, mem_bind=None, gpu_bind=None, nic_bind=None
        )

        assert c.profiling == m.Profiling(
            profile=False,
            cprofile=False,
            nvprof=False,
            nsys=False,
            nsys_targets="",
            nsys_extra=[],
        )

        assert c.logging == m.Logging(
            user_logging_levels=None, logdir=Path(), log_to_file=False
        )

        assert c.debugging == m.Debugging(
            gdb=False,
            cuda_gdb=False,
            memcheck=False,
            valgrind=False,
            freeze_on_error=False,
            gasnet_trace=False,
            spy=False,
        )

        assert c.info == m.Info(verbose=False, bind_detail=False)

        assert c.other == m.Other(
            timing=False,
            wrapper=[],
            wrapper_inner=[],
            module=None,
            dry_run=False,
            color=False,
        )

    def test_arg_conversions(self, mocker: MockerFixture) -> None:
        # This is kind of a dumb short-cut test, but if we believe that
        # object_to_dataclass works as advertised, then this test ensures that
        # it is being used for all the sub-configs that it should be used for

        spy = mocker.spy(m, "object_to_dataclass")

        c = m.Config(["legate"])

        assert spy.call_count == 4
        spy.assert_has_calls(
            [
                call(c._args, m.Kernel),
                call(c._args, MultiNode),
                call(c._args, Core),
                call(c._args, Memory),
            ]
        )
