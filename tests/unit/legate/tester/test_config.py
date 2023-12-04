# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
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

import os
import shutil
from pathlib import Path, PurePath

import pytest
from pytest_mock import MockerFixture

from legate.tester import FEATURES, config as m, defaults
from legate.tester.args import PIN_OPTIONS, PinOptionsType
from legate.util import colors

REPO_TOP = Path(__file__).parents[4]


class TestConfig:
    def test_default_init(self) -> None:
        c = m.Config([])

        assert colors.ENABLED is False

        assert c.features == ("cpus",)

        assert c.examples is True
        assert c.integration is True
        assert c.unit is False
        assert c.files is None
        assert c.last_failed is False
        assert c.gtest_file is None
        assert c.gtest_tests == []
        assert c.test_root is None

        assert c.core.cpus == defaults.CPUS_PER_NODE
        assert c.core.gpus == defaults.GPUS_PER_NODE
        assert c.core.omps == defaults.OMPS_PER_NODE
        assert c.core.ompthreads == defaults.OMPTHREADS

        assert c.memory.sysmem == defaults.SYS_MEMORY_BUDGET
        assert c.memory.fbmem == defaults.GPU_MEMORY_BUDGET
        assert c.memory.numamem == defaults.NUMA_MEMORY_BUDGET

        assert c.multi_node.nodes == defaults.NODES
        assert c.multi_node.ranks_per_node == defaults.RANKS_PER_NODE
        assert c.multi_node.launcher == "none"
        assert c.multi_node.launcher_extra == []
        assert c.multi_node.mpi_output_filename is None

        assert c.execution.workers is None
        assert c.execution.timeout is None
        assert c.execution.gpu_delay == defaults.GPU_DELAY
        assert c.execution.bloat_factor == defaults.GPU_BLOAT_FACTOR
        assert c.execution.cpu_pin == "partial"

        assert c.info.debug is False
        assert c.info.verbose == 0

        assert c.other.dry_run is False
        assert c.other.cov_bin is None
        assert c.other.cov_args == "run -a --branch"
        assert c.other.cov_src_path is None
        assert c.other.legate_dir is None

        assert c.extra_args == []
        assert c.root_dir == PurePath(os.getcwd())
        assert c.dry_run is False
        assert c.legate_path == shutil.which("legate")

    def test_color_arg(self) -> None:
        m.Config(["test.py", "--color"])

        assert colors.ENABLED is True

    def test_files(self) -> None:
        c = m.Config(["test.py", "--files", "a", "b", "c"])
        assert c.files == ["a", "b", "c"]

    def test_last_failed(self) -> None:
        c = m.Config(["test.py", "--last-failed"])
        assert c.last_failed

    @pytest.mark.parametrize("feature", FEATURES)
    def test_env_features(
        self, monkeypatch: pytest.MonkeyPatch, feature: str
    ) -> None:
        monkeypatch.setenv(f"USE_{feature.upper()}", "1")

        # test default config
        c = m.Config([])
        assert set(c.features) == {feature}

        # also test with a --use value provided
        c = m.Config(["test.py", "--use", "cuda"])
        assert set(c.features) == {"cuda"}

    @pytest.mark.parametrize("feature", FEATURES)
    def test_cmd_features(self, feature: str) -> None:
        # test a single value
        c = m.Config(["test.py", "--use", feature])
        assert set(c.features) == {feature}

        # also test with multiple / duplication
        c = m.Config(["test.py", "--use", f"cpus,{feature}"])
        assert set(c.features) == {"cpus", feature}

    @pytest.mark.parametrize("opt", ("cpus", "gpus", "omps", "ompthreads"))
    def test_core_options(self, opt: str) -> None:
        c = m.Config(["test.py", f"--{opt}", "1234"])
        assert getattr(c.core, opt.replace("-", "_")) == 1234

    @pytest.mark.parametrize("opt", ("sysmem", "fbmem", "numamem"))
    def test_memory_options(self, opt: str) -> None:
        c = m.Config(["test.py", f"--{opt}", "1234"])
        assert getattr(c.memory, opt.replace("-", "_")) == 1234

    def test_gpu_delay(self) -> None:
        c = m.Config(["test.py", "--gpu-delay", "1234"])
        assert c.execution.gpu_delay == 1234

    @pytest.mark.parametrize("value", PIN_OPTIONS)
    def test_cpu_pin(self, value: PinOptionsType) -> None:
        c = m.Config(["test.py", "--cpu-pin", value])
        assert c.execution.cpu_pin == value

    def test_workers(self) -> None:
        c = m.Config(["test.py", "-j", "1234"])
        assert c.execution.workers == 1234

    def test_timeout(self) -> None:
        c = m.Config(["test.py", "--timeout", "10"])
        assert c.execution.timeout == 10

    def test_debug(self) -> None:
        c = m.Config(["test.py", "--debug"])
        assert c.info.debug is True

    def test_dry_run(self) -> None:
        c = m.Config(["test.py", "--dry-run"])
        assert c.other.dry_run is True

    @pytest.mark.parametrize("arg", ("-v", "--verbose"))
    def test_verbose1(self, arg: str) -> None:
        c = m.Config(["test.py", arg])
        assert c.info.verbose == 1

    def test_verbose2(self) -> None:
        c = m.Config(["test.py", "-vv"])
        assert c.info.verbose == 2

    @pytest.mark.parametrize("arg", ("-C", "--directory"))
    def test_test_root(self, arg: str) -> None:
        c = m.Config(["test.py", arg, "some/path"])
        assert c.test_root == "some/path"

    def test_legate_dir(self) -> None:
        c = m.Config([])
        assert c.other.legate_dir is None
        assert c.legate_path == shutil.which("legate")
        assert c._legate_source == "install"

    def test_cmd_legate_dir_good(self) -> None:
        legate_dir = Path("/usr/local")
        c = m.Config(["test.py", "--legate", str(legate_dir)])
        assert c.other.legate_dir == legate_dir
        assert c.legate_path == str(legate_dir / "bin" / "legate")
        assert c._legate_source == "cmd"

    def test_env_legate_dir_good(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        legate_dir = Path("/usr/local")
        monkeypatch.setenv("LEGATE_DIR", str(legate_dir))
        c = m.Config([])
        assert c.other.legate_dir == legate_dir
        assert c.legate_path == str(legate_dir / "bin" / "legate")
        assert c._legate_source == "env"

    def test_extra_args(self) -> None:
        extra = ["-foo", "--bar", "--baz", "10"]
        c = m.Config(["test.py"] + extra)
        assert c.extra_args == extra

        # also test with --files since that option collects arguments
        c = m.Config(["test.py", "--files", "a", "b"] + extra)
        assert c.extra_args == extra
        c = m.Config(["test.py"] + extra + ["--files", "a", "b"])
        assert c.extra_args == extra

    def test_cov_args(self) -> None:
        cov_args = ["--cov-args", "run -a"]
        c = m.Config(["test.py"] + cov_args)
        assert c.other.cov_args == "run -a"


class Test_test_files:
    # first two tests are too sensitive to actual repo state and run location

    @pytest.mark.skip
    def test_basic(self) -> None:
        c = m.Config(["test.py", "--root-dir", str(REPO_TOP)])

        assert len(c.test_files) > 0
        assert any("examples" in str(x) for x in c.test_files)
        assert any("integration" in str(x) for x in c.test_files)

        assert not any("unit" in str(x) for x in c.test_files)

    @pytest.mark.skip
    def test_unit(self) -> None:
        c = m.Config(["test.py", "--unit", "--root-dir", str(REPO_TOP)])
        assert len(c.test_files) > 0
        assert any("unit" in str(x) for x in c.test_files)

    def test_error(self) -> None:
        c = m.Config(["test.py", "--files", "a", "b", "--last-failed"])
        with pytest.raises(RuntimeError):
            c.test_files

    @pytest.mark.parametrize("data", ("", " ", "\n", " \n "))
    def test_last_failed_empty(self, mocker: MockerFixture, data: str) -> None:
        mock_last_failed = mocker.mock_open(read_data=data)
        mocker.patch("builtins.open", mock_last_failed)
        c1 = m.Config(
            ["test.py", "--last-failed", "--root-dir", str(REPO_TOP)]
        )
        c2 = m.Config(["test.py", "--root-dir", str(REPO_TOP)])
        assert c1.test_files == c2.test_files

    def test_last_failed(self, mocker: MockerFixture) -> None:
        mock_last_failed = mocker.mock_open(read_data="\nfoo\nbar\nbaz\n")
        mocker.patch("builtins.open", mock_last_failed)
        c = m.Config(["test.py", "--last-failed", "--root-dir", str(REPO_TOP)])
        assert c.test_files == (Path("foo"), Path("bar"), Path("baz"))
