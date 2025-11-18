# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import operator
import functools
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import call

import pytest

import legate.driver.config as m
from legate.util import defaults
from legate.util.types import DataclassMixin

from ...util import powerset

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


class TestMultiNode:
    def test_fields(self) -> None:
        assert set(m.MultiNode.__dataclass_fields__) == {
            "nodes",
            "ranks_per_node",
            "launcher",
            "launcher_extra",
        }

    def test_mixin(self) -> None:
        assert issubclass(m.MultiNode, DataclassMixin)

    @pytest.mark.parametrize(
        "extra",
        (["a"], ["a", "b c"], ["a", "b c", "d e"], ["a", "b c", "d e", "f"]),
    )
    def test_launcher_extra_fixup_basic(self, extra: list[str]) -> None:
        mn = m.MultiNode(
            nodes=1, ranks_per_node=1, launcher="mpirun", launcher_extra=extra
        )
        assert mn.launcher_extra == functools.reduce(
            operator.iadd, (x.split() for x in extra), []
        )

    def test_launcher_extra_fixup_complex(self) -> None:
        mn = m.MultiNode(
            nodes=1,
            ranks_per_node=1,
            launcher="mpirun",
            launcher_extra=[
                "-H g0002,g0002 -X SOMEENV --fork",
                "-bind-to none",
            ],
        )
        assert mn.launcher_extra == [
            "-H",
            "g0002,g0002",
            "-X",
            "SOMEENV",
            "--fork",
            "-bind-to",
            "none",
        ]

    def test_launcher_extra_fixup_quoted(self) -> None:
        mn = m.MultiNode(
            nodes=1,
            ranks_per_node=1,
            launcher="mpirun",
            launcher_extra=["-f 'some path with spaces/foo.txt'"],
        )
        assert mn.launcher_extra == ["-f", "some path with spaces/foo.txt"]


class TestBinding:
    def test_fields(self) -> None:
        assert set(m.Binding.__dataclass_fields__) == {
            "cpu_bind",
            "mem_bind",
            "gpu_bind",
            "nic_bind",
        }

    def test_mixin(self) -> None:
        assert issubclass(m.Binding, DataclassMixin)


class TestCore:
    def test_fields(self) -> None:
        assert set(m.Core.__dataclass_fields__) == {
            "cpus",
            "gpus",
            "omps",
            "ompthreads",
            "utility",
        }

    def test_mixin(self) -> None:
        assert issubclass(m.Core, DataclassMixin)


class TestMemory:
    def test_fields(self) -> None:
        assert set(m.Memory.__dataclass_fields__) == {
            "sysmem",
            "numamem",
            "fbmem",
            "zcmem",
            "regmem",
            "max_exception_size",
            "min_cpu_chunk",
            "min_gpu_chunk",
            "min_omp_chunk",
            "field_reuse_fraction",
            "field_reuse_frequency",
            "consensus",
        }

    def test_mixin(self) -> None:
        assert issubclass(m.Memory, DataclassMixin)


class TestProfiling:
    def test_fields(self) -> None:
        assert set(m.Profiling.__dataclass_fields__) == {
            "profile",
            "profile_name",
            "provenance",
            "cprofile",
            "nvprof",
            "nsys",
            "nsys_extra",
        }

    def test_mixin(self) -> None:
        assert issubclass(m.Profiling, DataclassMixin)

    @pytest.mark.parametrize(
        "extra",
        (["a"], ["a", "b c"], ["a", "b c", "d e"], ["a", "b c", "d e", "f"]),
    )
    def test_nsys_extra_fixup_basic(self, extra: list[str]) -> None:
        p = m.Profiling(
            profile=True,
            profile_name=None,
            provenance=None,
            cprofile=True,
            nvprof=True,
            nsys=True,
            nsys_extra=extra,
        )
        assert p.nsys_extra == functools.reduce(
            operator.iadd, (x.split() for x in extra), []
        )

    def test_nsys_extra_fixup_complex(self) -> None:
        p = m.Profiling(
            profile=True,
            profile_name="foo",
            provenance=None,
            cprofile=True,
            nvprof=True,
            nsys=True,
            nsys_extra=["-H g0002,g0002 -X SOMEENV --fork", "-bind-to none"],
        )
        assert p.nsys_extra == [
            "-H",
            "g0002,g0002",
            "-X",
            "SOMEENV",
            "--fork",
            "-bind-to",
            "none",
        ]

    def test_nsys_extra_fixup_quoted(self) -> None:
        p = m.Profiling(
            profile=True,
            profile_name=None,
            provenance=None,
            cprofile=True,
            nvprof=True,
            nsys=True,
            nsys_extra=["-f 'some path with spaces/foo.txt'"],
        )
        assert p.nsys_extra == ["-f", "some path with spaces/foo.txt"]


class TestLogging:
    def test_fields(self) -> None:
        assert set(m.Logging.__dataclass_fields__) == {
            "user_logging_levels",
            "logdir",
            "log_to_file",
        }

    def test_mixin(self) -> None:
        assert issubclass(m.Logging, DataclassMixin)


class TestDebugging:
    def test_fields(self) -> None:
        assert set(m.Debugging.__dataclass_fields__) == {
            "gdb",
            "cuda_gdb",
            "memcheck",
            "valgrind",
            "freeze_on_error",
            "gasnet_trace",
        }

    def test_mixin(self) -> None:
        assert issubclass(m.Debugging, DataclassMixin)


class TestInfo:
    def test_fields(self) -> None:
        assert set(m.Info.__dataclass_fields__) == {"verbose", "bind_detail"}

    def test_mixin(self) -> None:
        assert issubclass(m.Info, DataclassMixin)


class TestOther:
    def test_fields(self) -> None:
        assert set(m.Other.__dataclass_fields__) == {
            "auto_config",
            "show_config",
            "show_memory_usage",
            "show_progress",
            "timing",
            "wrapper",
            "wrapper_inner",
            "module",
            "dry_run",
            "color",
            "window_size",
            "warmup_nccl",
            "disable_mpi",
            "inline_task_launch",
            "single_controller_execution",
            "io_use_vfd_gds",
            "experimental_copy_path",
        }

    def test_mixin(self) -> None:
        assert issubclass(m.Other, DataclassMixin)


class TestConfig:
    def test_default_init(self) -> None:
        # Note this test does not clear the environment. Default values from
        # the defaults module can depend on the environment, but what matters
        # is that the generated config matches those values, whatever they are.

        c = m.Config(["legate"])

        assert c.multi_node == m.MultiNode(
            nodes=defaults.LEGATE_NODES,
            ranks_per_node=defaults.LEGATE_RANKS_PER_NODE,
            launcher="none",
            launcher_extra=[],
        )
        assert c.binding == m.Binding(
            cpu_bind=None, mem_bind=None, gpu_bind=None, nic_bind=None
        )
        assert c.core == m.Core(
            cpus=None, gpus=None, omps=None, ompthreads=None, utility=None
        )

        assert c.memory == m.Memory(
            sysmem=None,
            numamem=None,
            fbmem=None,
            zcmem=None,
            regmem=None,
            max_exception_size=None,
            min_cpu_chunk=None,
            min_gpu_chunk=None,
            min_omp_chunk=None,
            field_reuse_fraction=None,
            field_reuse_frequency=None,
            consensus=False,
        )

        assert c.profiling == m.Profiling(
            profile=False,
            profile_name=None,
            provenance=None,
            cprofile=False,
            nvprof=False,
            nsys=False,
            nsys_extra=[],
        )

        assert c.logging == m.Logging(
            user_logging_levels=None, logdir=Path.cwd(), log_to_file=False
        )

        assert c.debugging == m.Debugging(
            gdb=False,
            cuda_gdb=False,
            memcheck=False,
            valgrind=False,
            freeze_on_error=False,
            gasnet_trace=False,
        )

        assert c.info == m.Info(verbose=False, bind_detail=False)

        assert c.other == m.Other(
            auto_config=False,
            show_config=False,
            show_memory_usage=False,
            show_progress=False,
            timing=False,
            wrapper=[],
            wrapper_inner=[],
            module=None,
            dry_run=False,
            color=False,
            window_size=None,
            warmup_nccl=False,
            disable_mpi=False,
            inline_task_launch=False,
            single_controller_execution=False,
            io_use_vfd_gds=False,
            experimental_copy_path=False,
        )

    def test_arg_conversions(self, mocker: MockerFixture) -> None:
        # This is kind of a dumb short-cut test, but if we believe that
        # object_to_dataclass works as advertised, then this test ensures that
        # it is being used for all the sub-configs that it should be used for

        spy = mocker.spy(m, "object_to_dataclass")

        c = m.Config(["legate"])

        assert spy.call_count == 9
        spy.assert_has_calls(
            [
                call(c._args, m.MultiNode),
                call(c._args, m.Binding),
                call(c._args, m.Core),
                call(c._args, m.Memory),
                call(c._args, m.Profiling),
                call(c._args, m.Logging),
                call(c._args, m.Debugging),
                call(c._args, m.Info),
                call(c._args, m.Other),
            ]
        )

    # maybe this is overkill but this is literally the point where the user's
    # own script makes contact with legate, so let's make extra sure that that
    # ingest succeeds over a very wide range of command line combinations (one
    # option from most sub-configs)
    @pytest.mark.parametrize(
        "args", powerset(("--gdb", "--profile", "--cprofile"))
    )
    def test_user_opts(self, args: tuple[str, ...]) -> None:
        c = m.Config(["legate", *list(args), "foo.py", "-a", "1"])

        assert c.user_opts == ("-a", "1")
        assert c.user_program == "foo.py"

    USER_OPTS: tuple[list[str], ...] = (
        [],
        ["-a"],
        ["-a", "-b", "1", "--long"],
        ["--cpus=2"],
    )

    @pytest.mark.parametrize("opts", USER_OPTS)
    def test_exec_run_mode_with_prog_with_module(
        self, opts: list[str]
    ) -> None:
        with pytest.raises(RuntimeError):
            m.Config(
                [
                    "legate",
                    "--run-mode",
                    "exec",
                    "--module",
                    "mod",
                    "prog",
                    *opts,
                ]
            )

    @pytest.mark.parametrize("opts", USER_OPTS)
    def test_exec_run_mode_with_prog_no_module(self, opts: list[str]) -> None:
        c = m.Config(["legate", "--run-mode", "exec", "prog", *opts])

        assert c.user_opts == tuple(opts)
        assert c.user_program == "prog"
        assert c.other.module is None
        assert c.run_mode == "exec"
        assert not c.console

    @pytest.mark.parametrize("opts", USER_OPTS)
    def test_python_run_mode_with_prog_with_module(
        self, opts: list[str]
    ) -> None:
        c = m.Config(
            [
                "legate",
                "--run-mode",
                "python",
                "--module",
                "mod",
                "prog",
                *opts,
            ]
        )

        assert c.user_opts == ()
        assert c.user_program is None
        assert c.other.module == ["mod", "prog", *opts]
        assert c.run_mode == "python"
        assert not c.console

    @pytest.mark.parametrize("opts", USER_OPTS)
    def test_python_run_mode_with_prog_no_module(
        self, opts: list[str]
    ) -> None:
        c = m.Config(["legate", "--run-mode", "python", "prog", *opts])

        assert c.user_opts == tuple(opts)
        assert c.user_program == "prog"
        assert c.other.module is None
        assert c.run_mode == "python"
        assert not c.console

    def test_python_run_mode_no_prog_with_module(self) -> None:
        c = m.Config(["legate", "--run-mode", "python", "--module", "mod"])

        assert c.user_opts == ()
        assert c.user_program is None
        assert c.other.module == ["mod"]
        assert c.run_mode == "python"
        assert not c.console

    def test_python_run_mode_no_prog_no_module(self) -> None:
        c = m.Config(["legate", "--run-mode", "python"])

        assert c.user_opts == ()
        assert c.user_program is None
        assert c.other.module is None
        assert c.run_mode == "python"
        assert c.console

    @pytest.mark.parametrize("opts", USER_OPTS)
    def test_default_run_mode_with_script_with_module(
        self, opts: list[str]
    ) -> None:
        c = m.Config(
            ["legate", "--gpus", "2", "--module", "mod", "script.py", *opts]
        )

        assert c.user_opts == ()
        assert c.user_program is None
        assert c.other.module == ["mod", "script.py", *opts]
        assert c.run_mode == "python"
        assert not c.console

    @pytest.mark.parametrize("opts", USER_OPTS)
    def test_default_run_mode_with_script_no_module(
        self, opts: list[str]
    ) -> None:
        c = m.Config(["legate", "--gpus", "2", "script.py", *opts])

        assert c.user_opts == tuple(opts)
        assert c.user_program == "script.py"
        assert c.other.module is None
        assert c.run_mode == "python"
        assert not c.console

    @pytest.mark.parametrize("opts", USER_OPTS)
    def test_default_run_mode_with_prog_with_modue(
        self, opts: list[str]
    ) -> None:
        c = m.Config(
            ["legate", "--gpus", "2", "--module", "mod", "prog", *opts]
        )

        assert c.user_opts == ()
        assert c.user_program is None
        assert c.other.module == ["mod", "prog", *opts]
        assert c.run_mode == "python"
        assert not c.console

    @pytest.mark.parametrize("opts", USER_OPTS)
    def test_default_run_mode_with_prog_no_modue(
        self, opts: list[str]
    ) -> None:
        c = m.Config(["legate", "--gpus", "2", "prog", *opts])

        assert c.user_opts == tuple(opts)
        assert c.user_program == "prog"
        assert c.other.module is None
        assert c.run_mode == "exec"
        assert not c.console

    def test_default_run_mode_no_prog_with_module(self) -> None:
        c = m.Config(["legate", "--module", "mod"])

        assert c.user_opts == ()
        assert c.user_program is None
        assert c.other.module == ["mod"]
        assert c.run_mode == "python"
        assert not c.console

    def test_default_run_mode_no_prog_no_module(self) -> None:
        c = m.Config(["legate"])

        assert c.user_opts == ()
        assert c.user_program is None
        assert c.other.module is None
        assert c.run_mode == "python"
        assert c.console

    def test_exec_run_mode_no_prog_with_module(self) -> None:
        with pytest.raises(RuntimeError):
            m.Config(["legate", "--run-mode", "exec", "--module", "mod"])

    def test_exec_run_mode_no_prog_no_module(self) -> None:
        with pytest.raises(RuntimeError):
            m.Config(["legate", "--run-mode", "exec"])
