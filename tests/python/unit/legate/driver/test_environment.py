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

from __future__ import annotations

import sys

import pytest

import legate.driver.environment as m

from .util import GenConfig


def test___all__() -> None:
    assert m.__all__ == ("ENV_PARTS_LEGATE",)


def test_ENV_PARTS_LEGATE() -> None:
    assert m.ENV_PARTS_LEGATE == (
        m.env_cpus,
        m.env_gpus,
        m.env_omps,
        m.env_ompthreads,
        m.env_utility,
        m.env_sysmem,
        m.env_numamem,
        m.env_fbmem,
        m.env_zcmem,
        m.env_regmem,
        m.env_log_levels,
        m.env_logdir,
        m.env_log_file,
        m.env_profile,
        m.env_spy,
        m.env_freeze_on_error,
    )


CORE_OPTS = ("cpus", "gpus", "omps", "ompthreads", "utility")
MEM_OPTS = ("sysmem", "numamem", "zcmem", "fbmem", "regmem")


@pytest.mark.parametrize("opt", CORE_OPTS)
class Test_core_opts:
    def test_default(self, genconfig: GenConfig, opt: str) -> None:
        config = genconfig([])
        func = getattr(m, f"env_{opt}")
        result = func(config)
        assert result == ()

    def test_value(self, genconfig: GenConfig, opt: str) -> None:
        config = genconfig([f"--{opt}", "10"])
        func = getattr(m, f"env_{opt}")
        result = func(config)
        assert result == (f"--{opt}", str(getattr(config.core, opt)))


@pytest.mark.parametrize("opt", MEM_OPTS)
class Test_mem_opts:
    def test_default(self, genconfig: GenConfig, opt: str) -> None:
        config = genconfig([])
        func = getattr(m, f"env_{opt}")
        result = func(config)
        assert result == ()

    def test_value(self, genconfig: GenConfig, opt: str) -> None:
        config = genconfig([f"--{opt}", "10"])
        func = getattr(m, f"env_{opt}")
        result = func(config)
        assert result == (f"--{opt}", str(getattr(config.memory, opt)))


class Test_log_levels:
    def test_default(self, genconfig: GenConfig) -> None:
        config = genconfig([])
        result = m.env_log_levels(config)
        assert result == ()

    def test_value(self, genconfig: GenConfig) -> None:
        config = genconfig(["--logging", "10"])
        result = m.env_log_levels(config)
        assert result == ("--logging", str(config.logging.user_logging_levels))


class Test_logdir:
    def test_default(self, genconfig: GenConfig) -> None:
        config = genconfig([])
        result = m.env_logdir(config)
        assert result == ("--logdir", str(config.logging.logdir))

    def test_value(self, genconfig: GenConfig) -> None:
        config = genconfig(["--logdir", "/foo/bar"])
        result = m.env_logdir(config)
        assert result == ("--logdir", str(config.logging.logdir))


class Test_log_file:
    def test_default(self, genconfig: GenConfig) -> None:
        config = genconfig([])
        result = m.env_log_file(config)
        assert result == ()

    def test_value(self, genconfig: GenConfig) -> None:
        config = genconfig(["--log-to-file"])
        result = m.env_log_file(config)
        assert result == ("--log-to-file",)


class Test_profile:
    def test_default(self, genconfig: GenConfig) -> None:
        config = genconfig([])
        result = m.env_profile(config)
        assert result == ()

    def test_value(self, genconfig: GenConfig) -> None:
        config = genconfig(["--profile"])
        result = m.env_profile(config)
        assert result == ("--profile",)


class Test_spy:
    def test_default(self, genconfig: GenConfig) -> None:
        config = genconfig([])
        result = m.env_spy(config)
        assert result == ()

    def test_value(self, genconfig: GenConfig) -> None:
        config = genconfig(["--spy"])
        result = m.env_spy(config)
        assert result == ("--spy",)


class Test_freeze_on_error:
    def test_default(self, genconfig: GenConfig) -> None:
        config = genconfig([])
        result = m.env_freeze_on_error(config)
        assert result == ()

    def test_value(self, genconfig: GenConfig) -> None:
        config = genconfig(["--freeze-on-error"])
        result = m.env_freeze_on_error(config)
        assert result == ("--freeze-on-error",)


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
