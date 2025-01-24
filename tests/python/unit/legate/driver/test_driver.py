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

import sys
from shlex import quote
from typing import TYPE_CHECKING

from rich.console import Console

import pytest

import legate.driver.driver as m
from legate import install_info
from legate.driver.command import CMD_PARTS_PYTHON
from legate.driver.config import Config
from legate.driver.launcher import RANK_ENV_VARS, Launcher
from legate.util.shared_args import LAUNCHERS
from legate.util.system import System

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

    from legate.util.types import LauncherType

    from ...util import Capsys
    from .util import GenConfig

SYSTEM = System()

CONSOLE = Console(color_system=None, soft_wrap=True)


class TestDriver:
    @pytest.mark.parametrize("launch", LAUNCHERS)
    def test_init(self, genconfig: GenConfig, launch: LauncherType) -> None:
        config = genconfig(["--launcher", launch])

        driver = m.LegateDriver(config, SYSTEM)

        assert driver.config is config
        assert driver.system is SYSTEM
        assert driver.launcher == Launcher.create(config, SYSTEM)

    @pytest.mark.parametrize("launch", LAUNCHERS)
    def test_cmd(self, genconfig: GenConfig, launch: LauncherType) -> None:
        config = genconfig(["--launcher", launch])

        driver = m.LegateDriver(config, SYSTEM)

        parts = (
            part(config, SYSTEM, driver.launcher) for part in CMD_PARTS_PYTHON
        )
        expected_cmd = driver.launcher.cmd + sum(parts, ())

        assert driver.cmd == expected_cmd

    @pytest.mark.parametrize("launch", LAUNCHERS)
    def test_env(self, genconfig: GenConfig, launch: LauncherType) -> None:
        config = genconfig(["--launcher", launch])

        system = System()

        # if someone runs the test as "legate -m pytest ..." then that first
        # legate invocation will modify the system env that subprocesses see
        if "LEGATE_CONFIG" in system.env:
            del system.env["LEGATE_CONFIG"]

        driver = m.LegateDriver(config, system)
        env = driver.env

        assert "LEGATE_CONFIG" in env
        del env["LEGATE_CONFIG"]

        assert env == driver.launcher.env

    @pytest.mark.parametrize("launch", LAUNCHERS)
    def test_custom_env_vars(
        self, genconfig: GenConfig, launch: LauncherType
    ) -> None:
        config = genconfig(["--launcher", launch])

        driver = m.LegateDriver(config, SYSTEM)

        assert driver.custom_env_vars == {
            "LEGATE_CONFIG",
            *driver.launcher.custom_env_vars,
        }

    @pytest.mark.parametrize("launch", LAUNCHERS)
    def test_dry_run(
        self, genconfig: GenConfig, mocker: MockerFixture, launch: LauncherType
    ) -> None:
        config = genconfig(["--launcher", launch, "--dry-run"])
        driver = m.LegateDriver(config, SYSTEM)

        mock_Popen = mocker.patch.object(m, "Popen")

        driver.run()

        mock_Popen.assert_not_called()

    @pytest.mark.parametrize("launch", LAUNCHERS)
    def test_run(
        self, genconfig: GenConfig, mocker: MockerFixture, launch: LauncherType
    ) -> None:
        config = genconfig(["--launcher", launch])
        driver = m.LegateDriver(config, SYSTEM)

        mock_Popen = mocker.patch.object(m, "Popen")

        driver.run()

        mock_Popen.assert_called_once_with(
            driver.cmd,
            env=driver.env,
            start_new_session=True,
            stdin=sys.stdin,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

    # skip simple launcher for this test
    @pytest.mark.parametrize("launch", ("mpirun", "jsrun", "srun"))
    def test_run_bad(
        self, genconfig: GenConfig, mocker: MockerFixture, launch: LauncherType
    ) -> None:
        config = genconfig(
            ["--launcher", launch, "--nodes", "2"], fake_module=None
        )
        driver = m.LegateDriver(config, SYSTEM)

        with pytest.raises(
            RuntimeError, match="Cannot start console with more than one node."
        ):
            driver.run()

    @pytest.mark.parametrize("launch", LAUNCHERS)
    def test_format_verbose(
        self, capsys: Capsys, genconfig: GenConfig, launch: LauncherType
    ) -> None:
        # set --dry-run to avoid needing to mock anything
        config = genconfig(["--launcher", launch, "--verbose", "--dry-run"])
        driver = m.LegateDriver(config, SYSTEM)

        driver.run()

        run_out = capsys.readouterr()[0].strip()

        with CONSOLE.capture() as capture:
            CONSOLE.print(m.format_verbose(driver.system, driver), end="")
        pv_out = capture.get()

        assert pv_out.strip() in run_out.strip()

    @pytest.mark.parametrize("rank_var", RANK_ENV_VARS)
    def test_verbose_nonzero_rank_id(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: Capsys,
        genconfig: GenConfig,
        rank_var: str,
    ) -> None:
        for name in RANK_ENV_VARS:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv(name, "1")
        monkeypatch.setattr(install_info, "networks", ["ucx"])

        # set --dry-run to avoid needing to mock anything
        config = genconfig(
            ["--launcher", "none", "--verbose", "--dry-run"], multi_rank=(2, 2)
        )
        system = System()
        driver = m.LegateDriver(config, system)

        driver.run()

        run_out = capsys.readouterr()[0].strip()

        with CONSOLE.capture() as capture:
            CONSOLE.print(m.format_verbose(system, driver), end="")
        pv_out = capture.get()

        assert pv_out not in run_out

    def test_module(self, genconfig: GenConfig, mocker: MockerFixture) -> None:
        config = genconfig(["--module", "some_mod"])
        driver = m.LegateDriver(config, SYSTEM)

        assert driver.cmd[-3:-1] == ("-m", "some_mod")


class Test_format_verbose:
    def test_system_only(self) -> None:
        system = System()

        text = m.format_verbose(system)
        with CONSOLE.capture() as capture:
            CONSOLE.print(text, end="")
        scrubbed = capture.get()

        assert "Legate paths" in scrubbed

        assert "Legion paths" in scrubbed

    def test_system_and_driver(self, capsys: Capsys) -> None:
        config = Config(["legate"])
        system = System()
        driver = m.LegateDriver(config, system)

        text = m.format_verbose(system, driver)
        with CONSOLE.capture() as capture:
            CONSOLE.print(text, end="")
        scrubbed = capture.get()

        assert "Legate paths" in scrubbed

        assert "Legion paths" in scrubbed

        assert "Command" in scrubbed
        assert f"{' '.join(quote(t) for t in driver.cmd)}" in scrubbed

        assert "Customized Environment" in scrubbed
        for k in driver.custom_env_vars:
            assert f"{k}={quote(driver.env[k])}" in scrubbed


# Below are the command-line options and flags that are handled by
# handle_legate_args in /src/cpp/legate/runtime/detail/argument_parsing.cc

OPTS = (
    "--cpus",
    "--gpus",
    "--omps",
    "--ompthreads",
    "--utility",
    "--sysmem",
    "--numamem",
    "--fbmem",
    "--zcmem",
    "--regmem",
    "--logging",
    "--logdir",
)

FLAGS = ("--profile", "--spy", "--log-to-file", "--freeze-on-error")


class Test_LEGATE_CONFIG:
    @pytest.mark.parametrize("opt", OPTS)
    def test_arg_opt_propagation(self, genconfig: GenConfig, opt: str) -> None:
        config = genconfig([opt, "10"])
        driver = m.LegateDriver(config, SYSTEM)
        LC = driver.env["LEGATE_CONFIG"]
        assert f"{opt} 10" in LC

    @pytest.mark.parametrize("flag", FLAGS)
    def test_arg_flag_propagation(
        self, genconfig: GenConfig, flag: str
    ) -> None:
        config = genconfig([flag])
        driver = m.LegateDriver(config, SYSTEM)
        LC = driver.env["LEGATE_CONFIG"]
        assert flag in LC
