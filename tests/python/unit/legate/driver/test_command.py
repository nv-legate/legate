# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import argparse
from typing import TYPE_CHECKING

import pytest

import legate.driver.command as m
from legate import install_info
from legate.driver.launcher import RANK_ENV_VARS

if TYPE_CHECKING:
    from legate.util.types import LauncherType

    from ...util import Capsys
    from .util import GenObjs


def test___all__() -> None:
    assert m.__all__ == ("CMD_PARTS_EXEC", "CMD_PARTS_PYTHON")


def test_LEGATE_GLOBAL_RANK_SUBSTITUTION() -> None:
    assert m.LEGATE_GLOBAL_RANK_SUBSTITUTION == "%%LEGATE_GLOBAL_RANK%%"


def test_CMD_PARTS() -> None:
    assert (
        m.cmd_bind,
        m.cmd_wrapper,
        m.cmd_gdb,
        m.cmd_cuda_gdb,
        m.cmd_nvprof,
        m.cmd_nsys,
        m.cmd_memcheck,
        m.cmd_valgrind,
        m.cmd_wrapper_inner,
        m.cmd_user_program,
        m.cmd_user_opts,
    ) == m.CMD_PARTS_EXEC

    assert (
        m.cmd_bind,
        m.cmd_wrapper,
        m.cmd_gdb,
        m.cmd_cuda_gdb,
        m.cmd_nvprof,
        m.cmd_nsys,
        m.cmd_memcheck,
        m.cmd_valgrind,
        m.cmd_wrapper_inner,
        m.cmd_python,
        m.cmd_module,
        m.cmd_user_program,
        m.cmd_user_opts,
    ) == m.CMD_PARTS_PYTHON


class Test_cmd_bind:
    def test_default(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs([])

        result = m.cmd_bind(config, system, launcher)

        bind_sh = str(system.legate_paths.bind_sh_path)
        assert result == (bind_sh, "--launcher", "local", "--")

    def test_bind_detail(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs(["--bind-detail"])

        result = m.cmd_bind(config, system, launcher)

        bind_sh = str(system.legate_paths.bind_sh_path)
        assert result == (bind_sh, "--launcher", "local", "--debug", "--")

    @pytest.mark.parametrize("kind", ("cpu", "gpu", "mem", "nic"))
    def test_basic_local(self, genobjs: GenObjs, kind: str) -> None:
        config, system, launcher = genobjs([f"--{kind}-bind", "1"])

        result = m.cmd_bind(config, system, launcher)

        bind_sh = str(system.legate_paths.bind_sh_path)
        assert result == (
            bind_sh,
            "--launcher",
            "local",
            f"--{kind}s",
            "1",
            "--",
        )

    @pytest.mark.parametrize("launch", ("none", "mpirun", "jsrun", "srun"))
    def test_combo_local(self, genobjs: GenObjs, launch: LauncherType) -> None:
        all_binds = [
            "--cpu-bind",
            "1",
            "--gpu-bind",
            "1",
            "--nic-bind",
            "1",
            "--mem-bind",
            "1",
            "--launcher",
            launch,
        ]
        config, system, launcher = genobjs(all_binds)

        result = m.cmd_bind(config, system, launcher)

        bind_sh = str(system.legate_paths.bind_sh_path)

        assert result[:3] == (
            bind_sh,
            "--launcher",
            "local" if launch == "none" else launch,
        )

        # TODO(jfaibussowit)
        # Replace below with itertools.batched once we hit python 3.12
        def pairwise(iterable: tuple[str, ...]) -> list[tuple[str, str]]:
            # pairwise('ABCDEF') -> AB CD EF
            from itertools import islice  # noqa: PLC0415

            iterator = iter(iterable)
            ret = []
            while batch := tuple(islice(iterator, 2)):
                if len(batch) != 2:
                    m = (
                        f"batched(): incomplete batch: {batch} "
                        f"(from {iterable})"
                    )
                    raise ValueError(m)
                ret.append(batch)
            return ret

        assert result[-1] == "--"

        for name, binding in pairwise(
            result[
                3:
                # -1 here because we don't want to include the "--" in the
                # batch
                -1
            ]
        ):
            assert f"{name} {binding}" in "--cpus 1 --gpus 1 --nics 1 --mems 1"

    @pytest.mark.parametrize("launch", ("none", "mpirun", "jsrun", "srun"))
    @pytest.mark.parametrize("rank_var", RANK_ENV_VARS)
    @pytest.mark.parametrize("kind", ("cpu", "gpu", "mem", "nic"))
    def test_ranks_good(
        self,
        genobjs: GenObjs,
        launch: LauncherType,
        kind: str,
        rank_var: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(install_info, "networks", ["ucx"])

        config, system, launcher = genobjs(
            [f"--{kind}-bind", "1/2", "--launcher", launch],
            multi_rank=(2, 2),
            rank_env={rank_var: "1"},
        )

        result = m.cmd_bind(config, system, launcher)

        launcher_arg = "auto" if launch == "none" else launch

        bind_sh = str(system.legate_paths.bind_sh_path)
        assert result == (
            bind_sh,
            "--launcher",
            launcher_arg,
            f"--{kind}s",
            "1/2",
            "--",
        )

    @pytest.mark.parametrize("binding", ("1", "1/2/3"))
    @pytest.mark.parametrize("rank_var", RANK_ENV_VARS)
    @pytest.mark.parametrize("kind", ("cpu", "gpu", "mem", "nic"))
    def test_ranks_bad(
        self,
        genobjs: GenObjs,
        binding: str,
        kind: str,
        rank_var: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(install_info, "networks", ["ucx"])

        config, system, launcher = genobjs(
            [f"--{kind}-bind", binding],
            multi_rank=(2, 2),
            rank_env={rank_var: "1"},
        )

        msg = (
            f"Number of groups in --{kind}-bind not equal to --ranks-per-node"
        )
        with pytest.raises(RuntimeError, match=msg):
            m.cmd_bind(config, system, launcher)

    @pytest.mark.parametrize("launch", ("none", "mpirun", "jsrun", "srun"))
    @pytest.mark.parametrize("rank_var", RANK_ENV_VARS)
    def test_no_networking_error(
        self,
        genobjs: GenObjs,
        launch: LauncherType,
        rank_var: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(install_info, "networks", [])

        config, system, launcher = genobjs(
            ["--launcher", launch], multi_rank=(2, 2), rank_env={rank_var: "1"}
        )

        msg = (
            "multi-rank run was requested, but Legate was not built with "
            "networking support"
        )
        with pytest.raises(RuntimeError, match=msg):
            m.cmd_bind(config, system, launcher)


class Test_cmd_gdb:
    MULTI_RANK_WARN = (
        "WARNING: Legate does not support gdb for multi-rank runs"
    )

    def test_default(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs([])

        result = m.cmd_gdb(config, system, launcher)

        assert result == ()

    @pytest.mark.parametrize("os", ("Darwin", "Linux"))
    def test_with_option(self, genobjs: GenObjs, os: str) -> None:
        config, system, launcher = genobjs(["--gdb"], os=os)

        result = m.cmd_gdb(config, system, launcher)

        debugger = ("lldb", "--") if os == "Darwin" else ("gdb", "--args")
        assert result == debugger

    @pytest.mark.parametrize("rank_var", RANK_ENV_VARS)
    @pytest.mark.parametrize("os", ("Darwin", "Linux"))
    def test_with_option_multi_rank(
        self, genobjs: GenObjs, capsys: Capsys, os: str, rank_var: str
    ) -> None:
        config, system, launcher = genobjs(
            ["--gdb"], multi_rank=(2, 2), rank_env={rank_var: "1"}, os=os
        )

        result = m.cmd_gdb(config, system, launcher)
        assert result == ()

        out, _ = capsys.readouterr()
        assert out.strip() == self.MULTI_RANK_WARN


class Test_cmd_cuda_gdb:
    MULTI_RANK_WARN = (
        "WARNING: Legate does not support cuda-gdb for multi-rank runs"
    )

    def test_default(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs([])

        result = m.cmd_cuda_gdb(config, system, launcher)

        assert result == ()

    @pytest.mark.parametrize("os", ("Darwin", "Linux"))
    def test_with_option(self, genobjs: GenObjs, os: str) -> None:
        config, system, launcher = genobjs(["--cuda-gdb"], os=os)

        result = m.cmd_cuda_gdb(config, system, launcher)

        assert result == ("cuda-gdb", "--args")

    @pytest.mark.parametrize("rank_var", RANK_ENV_VARS)
    @pytest.mark.parametrize("os", ("Darwin", "Linux"))
    def test_with_option_multi_rank(
        self, genobjs: GenObjs, capsys: Capsys, os: str, rank_var: str
    ) -> None:
        config, system, launcher = genobjs(
            ["--cuda-gdb"], multi_rank=(2, 2), rank_env={rank_var: "1"}, os=os
        )

        result = m.cmd_cuda_gdb(config, system, launcher)
        assert result == ()

        out, _ = capsys.readouterr()
        assert out.strip() == self.MULTI_RANK_WARN


class Test_cmd_nvprof:
    def test_default(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs([])

        result = m.cmd_nvprof(config, system, launcher)

        assert result == ()

    def test_with_option(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs(["--nvprof", "--logdir", "foo"])

        result = m.cmd_nvprof(config, system, launcher)

        log_path = str(
            config.logging.logdir
            / f"legate_{m.LEGATE_GLOBAL_RANK_SUBSTITUTION}.nvvp"
        )
        assert result == ("nvprof", "-o", log_path)

    @pytest.mark.parametrize("rank_var", RANK_ENV_VARS)
    @pytest.mark.parametrize("rank", ("0", "1", "2"))
    def test_multi_rank_no_launcher(
        self, genobjs: GenObjs, rank_var: str, rank: str
    ) -> None:
        config, system, launcher = genobjs(
            ["--nvprof", "--logdir", "foo"],
            multi_rank=(2, 2),
            rank_env={rank_var: rank},
        )

        result = m.cmd_nvprof(config, system, launcher)

        log_path = str(
            config.logging.logdir
            / f"legate_{m.LEGATE_GLOBAL_RANK_SUBSTITUTION}.nvvp"
        )
        assert result == ("nvprof", "-o", log_path)

    @pytest.mark.parametrize("launch", ("mpirun", "jsrun", "srun"))
    def test_multi_rank_with_launcher(
        self, genobjs: GenObjs, launch: str
    ) -> None:
        config, system, launcher = genobjs(
            ["--nvprof", "--logdir", "foo", "--launcher", launch],
            multi_rank=(2, 2),
        )

        result = m.cmd_nvprof(config, system, launcher)

        log_path = str(
            config.logging.logdir
            / f"legate_{m.LEGATE_GLOBAL_RANK_SUBSTITUTION}.nvvp"
        )
        assert result == ("nvprof", "-o", log_path)


class Test_cmd_nsys:
    def test_default(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs([])

        result = m.cmd_nsys(config, system, launcher)

        assert result == ()

    @pytest.mark.parametrize("rank_var", RANK_ENV_VARS)
    @pytest.mark.parametrize("rank", ("0", "1", "2"))
    def test_multi_rank_no_launcher(
        self, genobjs: GenObjs, rank_var: str, rank: str
    ) -> None:
        config, system, launcher = genobjs(
            ["--nsys", "--logdir", "foo"],
            multi_rank=(2, 2),
            rank_env={rank_var: rank},
        )

        result = m.cmd_nsys(config, system, launcher)

        log_path = str(
            config.logging.logdir
            / f"legate_{m.LEGATE_GLOBAL_RANK_SUBSTITUTION}"
        )
        assert result == ("nsys", "profile", "-o", log_path)

    @pytest.mark.parametrize(
        "nsys_extra",
        (
            ["--nsys-extra=--sample=cpu"],
            ["--nsys-extra", "--backtrace=lbr -s cpu"],
            ["--nsys-extra", "--sample cpu"],
            ["--nsys-extra", "-s cpu"],
            ["--nsys-extra=--sample", "--nsys-extra", "cpu"],
        ),
    )
    def test_explicit_sample(
        self, genobjs: GenObjs, nsys_extra: list[str]
    ) -> None:
        args = ["--nsys", *nsys_extra]
        config, system, launcher = genobjs(args)
        result = m.cmd_nsys(config, system, launcher)

        parser = argparse.ArgumentParser()
        parser.add_argument("-s", "--sample")
        parsed_args, _ = parser.parse_known_args(result)

        assert parsed_args.sample == "cpu"

    @pytest.mark.parametrize("launch", ("mpirun", "jsrun", "srun"))
    def test_multi_rank_with_launcher(
        self, genobjs: GenObjs, launch: str
    ) -> None:
        config, system, launcher = genobjs(
            ["--nsys", "--logdir", "foo", "--launcher", launch],
            multi_rank=(2, 2),
        )

        result = m.cmd_nsys(config, system, launcher)

        log_path = str(
            config.logging.logdir
            / f"legate_{m.LEGATE_GLOBAL_RANK_SUBSTITUTION}"
        )
        assert result == ("nsys", "profile", "-o", log_path)

    @pytest.mark.parametrize("rank_var", RANK_ENV_VARS)
    @pytest.mark.parametrize("rank", ("0", "1", "2"))
    def test_multi_rank_extra_no_s(
        self, genobjs: GenObjs, rank_var: str, rank: str
    ) -> None:
        config, system, launcher = genobjs(
            [
                "--nsys",
                "--logdir",
                "foo",
                "--nsys-extra",
                "a",
                "--nsys-extra",
                "b",
            ],
            multi_rank=(2, 2),
            rank_env={rank_var: rank},
        )

        result = m.cmd_nsys(config, system, launcher)

        log_path = str(
            config.logging.logdir
            / f"legate_{m.LEGATE_GLOBAL_RANK_SUBSTITUTION}"
        )
        assert result == ("nsys", "profile", "-o", log_path, "a", "b")

    @pytest.mark.parametrize("rank_var", RANK_ENV_VARS)
    @pytest.mark.parametrize("rank", ("0", "1", "2"))
    def test_multi_rank_extra_with_s(
        self, genobjs: GenObjs, rank_var: str, rank: str
    ) -> None:
        config, system, launcher = genobjs(
            [
                "--nsys",
                "--logdir",
                "foo",
                "--nsys-extra",
                "a",
                "--nsys-extra",
                "b",
                "--nsys-extra=-s",  # note have to use "=" format
                "--nsys-extra",
                "foo",
            ],
            multi_rank=(2, 2),
            rank_env={rank_var: rank},
        )

        result = m.cmd_nsys(config, system, launcher)

        log_path = str(
            config.logging.logdir
            / f"legate_{m.LEGATE_GLOBAL_RANK_SUBSTITUTION}"
        )
        assert result == (
            "nsys",
            "profile",
            "-o",
            log_path,
            "a",
            "b",
            "-s",
            "foo",
        )


class Test_cmd_memcheck:
    def test_default(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs([])

        result = m.cmd_memcheck(config, system, launcher)

        assert result == ()

    def test_with_option(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs(["--memcheck"])

        result = m.cmd_memcheck(config, system, launcher)

        assert result == ("compute-sanitizer",)


class Test_cmd_wrapper:
    def test_default(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs([])

        result = m.cmd_wrapper(config, system, launcher)

        assert result == ()

    def test_with_option(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs(
            ["--wrapper", "foo --bar 10 -s -baz=20"]
        )

        result = m.cmd_wrapper(config, system, launcher)

        assert result == ("foo", "--bar", "10", "-s", "-baz=20")

    def test_multi(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs(
            ["--wrapper", "foo --bar 10", "--wrapper", "baz -s"]
        )

        result = m.cmd_wrapper(config, system, launcher)

        assert result == ("foo", "--bar", "10", "baz", "-s")


class Test_cmd_wrapper_inner:
    def test_default(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs([])

        result = m.cmd_wrapper_inner(config, system, launcher)

        assert result == ()

    def test_with_option(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs(
            ["--wrapper-inner", "foo --bar 10 -s -baz=20"]
        )

        result = m.cmd_wrapper_inner(config, system, launcher)

        assert result == ("foo", "--bar", "10", "-s", "-baz=20")

    def test_multi(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs(
            ["--wrapper-inner", "foo --bar 10", "--wrapper-inner", "baz -s"]
        )

        result = m.cmd_wrapper_inner(config, system, launcher)

        assert result == ("foo", "--bar", "10", "baz", "-s")


class Test_cmd_valgrind:
    def test_default(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs([])

        result = m.cmd_valgrind(config, system, launcher)

        assert result == ()

    def test_with_option(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs(["--valgrind"])

        result = m.cmd_valgrind(config, system, launcher)

        assert result == ("valgrind",)


class Test_cmd_module:
    def test_default(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs([])

        result = m.cmd_module(config, system, launcher)

        assert result == ()

    def test_with_module(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs(["--module", "bar"])

        result = m.cmd_module(config, system, launcher)

        assert result == ("-m", "bar", "foo.py")

    def test_with_cprofile(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs(["--cprofile"])

        result = m.cmd_module(config, system, launcher)

        log_path = str(
            config.logging.logdir
            / f"legate_{m.LEGATE_GLOBAL_RANK_SUBSTITUTION}.cprof"
        )
        assert result == ("-m", "cProfile", "-o", log_path)

    def test_module_and_cprofile_error(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs(["--cprofile", "--module", "foo"])

        err = "Only one of --module or --cprofile may be used"
        with pytest.raises(ValueError, match=err):
            m.cmd_module(config, system, launcher)


class Test_cmd_user_opts:
    USER_OPTS: tuple[list[str], ...] = (
        [],
        ["foo"],
        ["foo.py"],
        ["foo.py", "10"],
        ["foo.py", "--baz", "10"],
    )

    @pytest.mark.parametrize("opts", USER_OPTS, ids=str)
    def test_basic(self, genobjs: GenObjs, opts: list[str]) -> None:
        config, system, launcher = genobjs(opts, fake_module=None)

        user_opts = m.cmd_user_opts(config, system, launcher)
        user_program = m.cmd_user_program(config, system, launcher)
        result = user_program + user_opts

        assert result == tuple(opts)

    @pytest.mark.parametrize("opts", USER_OPTS, ids=str)
    @pytest.mark.skipif(not install_info.use_cuda, reason="no CUDA support")
    def test_with_legate_opts(self, genobjs: GenObjs, opts: list[str]) -> None:
        args = ["--verbose", "--gpus", "2", *opts]
        config, system, launcher = genobjs(args, fake_module=None)

        user_opts = m.cmd_user_opts(config, system, launcher)
        user_program = m.cmd_user_program(config, system, launcher)
        result = user_program + user_opts

        assert result == tuple(opts)


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
