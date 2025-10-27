# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest

import legate.driver.environment as m

if TYPE_CHECKING:
    from .util import GenConfig


def test___all__() -> None:
    assert m.__all__ == ("ENV_PARTS_LEGATE",)


def test_ENV_PARTS_LEGATE() -> None:
    assert (
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
        m.env_max_exception_size,
        m.env_min_cpu_chunk,
        m.env_min_gpu_chunk,
        m.env_min_omp_chunk,
        m.env_field_reuse_fraction,
        m.env_field_reuse_frequency,
        m.env_consensus,
        m.env_log_levels,
        m.env_logdir,
        m.env_log_file,
        m.env_profile,
        m.env_profile_name,
        m.env_provenance,
        m.env_freeze_on_error,
        m.env_auto_config,
        m.env_show_config,
        m.env_show_memory_usage,
        m.env_show_progress,
        m.env_window_size,
        m.env_warmup_nccl,
        m.env_disable_mpi,
        m.env_inline_task_launch,
        m.env_io_use_vfd_gds,
        m.env_experimental_copy_path,
    ) == m.ENV_PARTS_LEGATE


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


class Test_profile_name:
    def test_default(self, genconfig: GenConfig) -> None:
        config = genconfig([])
        result = m.env_profile_name(config)
        assert result == ()

    def test_value(self, genconfig: GenConfig) -> None:
        config = genconfig(["--profile-name", "foo"])
        result = m.env_profile_name(config)
        assert result == ("--profile-name", "foo")


class Test_provenance:
    def test_default(self, genconfig: GenConfig) -> None:
        config = genconfig([])
        result = m.env_provenance(config)
        assert result == ()

    def test_default_with_profile(self, genconfig: GenConfig) -> None:
        config = genconfig(["--profile"])
        result = m.env_provenance(config)
        assert result == ("--provenance",)

    def test_value(self, genconfig: GenConfig) -> None:
        config = genconfig(["--provenance"])
        result = m.env_provenance(config)
        assert result == ("--provenance",)


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
