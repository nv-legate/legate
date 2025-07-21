# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import legate.driver as m
import legate.util.system
import legate.driver.config
import legate.driver.driver

if TYPE_CHECKING:
    import pytest
    from pytest_mock import MockerFixture

# main function shadows main module
# def test___all__() -> None:

# The main() function is very simple, this test just confirms that
# all the expected plumbing is hooked up as it is supposed to be


def test_main(mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    config_spy = mocker.spy(legate.driver.config.Config, "__init__")
    system_spy = mocker.spy(legate.util.system.System, "__init__")
    driver_spy = mocker.spy(legate.driver.driver.LegateDriver, "__init__")
    mocker.patch("legate.driver.driver.LegateDriver.run", return_value=123)
    mocker.patch.object(sys, "argv", ["/some/path", "bar"])

    # LEGATE_CONFIG should get spliced into the start of argv[1:]
    monkeypatch.setenv("LEGATE_CONFIG", "foo")

    result = m.main()

    assert config_spy.call_count == 1
    assert config_spy.call_args[0][1][1:] == ["foo", "bar"]
    assert config_spy.call_args[1] == {}

    assert system_spy.call_count == 1
    assert system_spy.call_args[0][1:] == ()
    assert system_spy.call_args[1] == {}

    assert driver_spy.call_count == 1
    assert len(driver_spy.call_args[0]) == 3
    assert isinstance(driver_spy.call_args[0][1], legate.driver.config.Config)
    assert isinstance(driver_spy.call_args[0][2], legate.util.system.System)
    assert driver_spy.call_args[1] == {}

    assert result == 123
