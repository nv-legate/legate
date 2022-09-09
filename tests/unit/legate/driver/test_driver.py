# Copyright 2021-2022 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from __future__ import annotations

from shlex import quote
from typing import Any

import pytest
from pytest_mock import MockerFixture

import legate.driver.driver as m
from legate.driver.args import LAUNCHERS
from legate.driver.command import CMD_PARTS
from legate.driver.launcher import Launcher
from legate.driver.system import System
from legate.driver.types import LauncherType
from legate.driver.ui import scrub

from util import Capsys

SYSTEM = System()


class TestDriver:
    @pytest.mark.parametrize("launch", LAUNCHERS)
    def test_init(self, genconfig: Any, launch: LauncherType) -> None:
        config = genconfig(["--launcher", launch])

        driver = m.Driver(config, SYSTEM)

        assert driver.config is config
        assert driver.system is SYSTEM
        assert driver.launcher == Launcher.create(config, SYSTEM)

    @pytest.mark.parametrize("launch", LAUNCHERS)
    def test_cmd(self, genconfig: Any, launch: LauncherType) -> None:
        config = genconfig(["--launcher", launch])

        driver = m.Driver(config, SYSTEM)

        parts = (part(config, SYSTEM, driver.launcher) for part in CMD_PARTS)
        expected_cmd = driver.launcher.cmd + sum(parts, ())

        assert driver.cmd == expected_cmd

    @pytest.mark.parametrize("launch", LAUNCHERS)
    def test_env(self, genconfig: Any, launch: LauncherType) -> None:
        config = genconfig(["--launcher", launch])

        driver = m.Driver(config, SYSTEM)

        assert driver.env == driver.launcher.env

    @pytest.mark.parametrize("launch", LAUNCHERS)
    def test_custom_env_vars(
        self, genconfig: Any, launch: LauncherType
    ) -> None:
        config = genconfig(["--launcher", launch])

        driver = m.Driver(config, SYSTEM)

        assert driver.custom_env_vars == driver.launcher.custom_env_vars

    @pytest.mark.parametrize("launch", LAUNCHERS)
    def test_dry_run(
        self, genconfig: Any, mocker: MockerFixture, launch: LauncherType
    ) -> None:
        config = genconfig(["--launcher", launch, "--dry-run"])
        driver = m.Driver(config, SYSTEM)
        mock_run = mocker.patch.object(m, "run")

        driver.run()

        mock_run.assert_not_called()

    @pytest.mark.parametrize("launch", LAUNCHERS)
    def test_run(
        self, genconfig: Any, mocker: MockerFixture, launch: LauncherType
    ) -> None:
        config = genconfig(["--launcher", launch])
        driver = m.Driver(config, SYSTEM)

        mocker.patch("legate.driver.driver.Driver._init_logging")
        mocker.patch("legate.driver.driver.Driver._process_logging")
        mock_run = mocker.patch.object(m, "run")

        driver.run()

        mock_run.assert_called_once_with(driver.cmd, env=driver.env)

    @pytest.mark.parametrize("launch", LAUNCHERS)
    def test_verbose(
        self, capsys: Capsys, genconfig: Any, mocker: MockerFixture, launch: LauncherType
    ) -> None:
        # set --dry-run to avoid needing to mock anything
        config = genconfig(["--launcher", launch, "--verbose", "--dry-run"])
        driver = m.Driver(config, SYSTEM)

        driver.run()

        out = scrub(capsys.readouterr()[0]).strip()

        assert out.startswith(f"{'--- Legion Python Configuration ':-<80}")
        assert "Legate paths:" in out
        for line in scrub(str(driver.system.legate_paths)).split():
            assert line in out

        assert "Legion paths:" in out
        for line in scrub(str(driver.system.legion_paths)).split():
            assert line in out

        assert "Command:" in out
        assert f"  {' '.join(quote(t) for t in driver.cmd)}" in out

        assert "Customized Environment:" in out
        for k in driver.custom_env_vars:
            assert f"{k}={driver.env[k].rstrip()}" in out

        assert out.endswith(f"\n{'-':-<80}")


