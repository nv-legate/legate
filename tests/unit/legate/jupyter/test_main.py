# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest
from pytest_mock import MockerFixture

import legate.jupyter as m

# main function shadows main module
# def test___all__() -> None:

# The main() function is very simple, this test just confirms that
# all the expected plumbing is hooked up as it is supposed to be


# TODO: this test with the fake argv path does not work for the way
# legate is installed in CI, so skip for now.
@pytest.mark.skip
def test_main(mocker: MockerFixture) -> None:
    import legate.driver.driver
    import legate.jupyter.config
    import legate.util.system

    config_spy = mocker.spy(legate.jupyter.config.Config, "__init__")
    system_spy = mocker.spy(legate.util.system.System, "__init__")
    driver_spy = mocker.spy(legate.driver.driver.LegateDriver, "__init__")
    generate_spy = mocker.spy(legate.jupyter.kernel, "generate_kernel_spec")
    install_mock = mocker.patch("legate.jupyter.kernel.install_kernel_spec")
    mocker.patch.object(
        sys, "argv", ["/some/path/legate-jupyter", "--name", "foo"]
    )

    m.main()

    assert config_spy.call_count == 1
    assert config_spy.call_args[0][1] == sys.argv
    assert config_spy.call_args[1] == {}

    assert system_spy.call_count == 1
    assert system_spy.call_args[0][1:] == ()
    assert system_spy.call_args[1] == {}

    assert driver_spy.call_count == 1
    assert len(driver_spy.call_args[0]) == 3
    assert isinstance(driver_spy.call_args[0][1], legate.jupyter.config.Config)
    assert isinstance(driver_spy.call_args[0][2], legate.util.system.System)
    assert driver_spy.call_args[1] == {}

    assert generate_spy.call_count == 1
    assert len(generate_spy.call_args[0]) == 2
    assert isinstance(
        generate_spy.call_args[0][0], legate.driver.driver.LegateDriver
    )
    assert isinstance(
        generate_spy.call_args[0][1], legate.jupyter.config.Config
    )
    assert generate_spy.call_args[1] == {}

    assert install_mock.call_count == 1
    assert install_mock.call_args[0][0] == generate_spy.spy_return
    assert isinstance(
        install_mock.call_args[0][1], legate.jupyter.config.Config
    )
    assert install_mock.call_args[1] == {}
