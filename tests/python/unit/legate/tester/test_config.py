# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Consolidate test configuration from command-line and environment."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from legate.tester import config as m, defaults
from legate.tester.args import PIN_OPTIONS, PinOptionsType
from legate.tester.project import Project

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

REPO_TOP = Path(__file__).parents[4]
PROJECT = Project()


class TestConfig:
    def test_default_init(self) -> None:
        c = m.Config([], project=PROJECT)

        assert c.features == ("cpus",)

        assert c.examples is True
        assert c.integration is True
        assert c.files is None
        assert c.last_failed is False
        # assert c.gtest_file is None
        # assert c.gtest_tests == []
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

        # best we can do with dynamic defaultS
        filename = c.multi_node.mpi_output_filename
        assert filename is None or str(filename).endswith("mpi_result")

        assert c.execution.workers is None
        assert c.execution.timeout == 5 * 60
        assert c.execution.gpu_delay == defaults.GPU_DELAY
        assert c.execution.bloat_factor == defaults.GPU_BLOAT_FACTOR
        assert c.execution.cpu_pin == "partial"

        assert c.info.debug is False
        assert c.info.verbose == 0

        assert c.other.dry_run is False
        assert c.other.cov_bin is None
        assert c.other.cov_args == "run -a --branch"
        assert c.other.cov_src_path is None
        assert c.other.legate_install_dir is None

        assert c.extra_args == []
        assert c.root_dir == Path.cwd()
        assert c.dry_run is False
        assert c.legate_path == shutil.which("legate")

    def test_files(self) -> None:
        c = m.Config(["test.py", "--files", "a", "b", "c"], project=PROJECT)
        assert c.files == ["a", "b", "c"]

    def test_last_failed(self) -> None:
        c = m.Config(["test.py", "--last-failed"], project=PROJECT)
        assert c.last_failed

    @pytest.mark.parametrize("feature", defaults.FEATURES)
    def test_env_features(
        self, monkeypatch: pytest.MonkeyPatch, feature: str
    ) -> None:
        monkeypatch.setenv(f"USE_{feature.upper()}", "1")

        # test default config
        c = m.Config([], project=PROJECT)
        assert set(c.features) == {feature}

        # also test with a --use value provided
        c = m.Config(["test.py", "--use", "cuda"], project=PROJECT)
        assert set(c.features) == {"cuda"}

    @pytest.mark.parametrize("feature", defaults.FEATURES)
    def test_cmd_features(self, feature: str) -> None:
        # test a single value
        c = m.Config(["test.py", "--use", feature], project=PROJECT)
        assert set(c.features) == {feature}

        # also test with multiple / duplication
        c = m.Config(["test.py", "--use", f"cpus,{feature}"], project=PROJECT)
        assert set(c.features) == {"cpus", feature}

    @pytest.mark.parametrize("opt", ("cpus", "gpus", "omps", "ompthreads"))
    def test_core_options(self, opt: str) -> None:
        c = m.Config(["test.py", f"--{opt}", "1234"], project=PROJECT)
        assert getattr(c.core, opt.replace("-", "_")) == 1234

    @pytest.mark.parametrize("opt", ("sysmem", "fbmem", "numamem"))
    def test_memory_options(self, opt: str) -> None:
        c = m.Config(["test.py", f"--{opt}", "1234"], project=PROJECT)
        assert getattr(c.memory, opt.replace("-", "_")) == 1234

    def test_gpu_delay(self) -> None:
        c = m.Config(["test.py", "--gpu-delay", "1234"], project=PROJECT)
        assert c.execution.gpu_delay == 1234

    @pytest.mark.parametrize("value", PIN_OPTIONS)
    def test_cpu_pin(self, value: PinOptionsType) -> None:
        c = m.Config(["test.py", "--cpu-pin", value], project=PROJECT)
        assert c.execution.cpu_pin == value

    def test_workers(self) -> None:
        c = m.Config(["test.py", "-j", "1234"], project=PROJECT)
        assert c.execution.workers == 1234

    def test_timeout(self) -> None:
        c = m.Config(["test.py", "--timeout", "10"], project=PROJECT)
        assert c.execution.timeout == 10

    def test_debug(self) -> None:
        c = m.Config(["test.py", "--debug"], project=PROJECT)
        assert c.info.debug is True

    def test_dry_run(self) -> None:
        c = m.Config(["test.py", "--dry-run"], project=PROJECT)
        assert c.other.dry_run is True

    @pytest.mark.parametrize("arg", ("-v", "--verbose"))
    def test_verbose1(self, arg: str) -> None:
        c = m.Config(["test.py", arg], project=PROJECT)
        assert c.info.verbose == 1

    def test_verbose2(self) -> None:
        c = m.Config(["test.py", "-vv"], project=PROJECT)
        assert c.info.verbose == 2

    @pytest.mark.parametrize("arg", ("-C", "--directory"))
    def test_test_root(self, arg: str) -> None:
        c = m.Config(["test.py", arg, "some/path"], project=PROJECT)
        assert c.test_root == "some/path"

    def test_legate_install_dir(self) -> None:
        c = m.Config([], project=PROJECT)
        assert c.other.legate_install_dir is None
        assert c.legate_path == shutil.which("legate")
        assert c._legate_source == "install"

    def test_cmd_legate_install_dir_good(self) -> None:
        legate_install_dir = Path("/usr/local")
        c = m.Config(
            ["test.py", "--legate", str(legate_install_dir)], project=PROJECT
        )
        assert c.other.legate_install_dir == legate_install_dir
        assert c.legate_path == str(legate_install_dir / "bin" / "legate")
        assert c._legate_source == "cmd"

    def test_env_legate_install_dir_good(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        legate_install_dir = Path("/usr/local")
        monkeypatch.setenv("LEGATE_INSTALL_DIR", str(legate_install_dir))
        c = m.Config([], project=PROJECT)
        assert c.other.legate_install_dir == legate_install_dir
        assert c.legate_path == str(legate_install_dir / "bin" / "legate")
        assert c._legate_source == "env"

    def test_extra_args(self) -> None:
        extra = ["-foo", "--bar", "--baz", "10"]
        c = m.Config(["test.py", *extra], project=PROJECT)
        assert c.extra_args == extra

        # also test with --files since that option collects arguments
        c = m.Config(["test.py", "--files", "a", "b", *extra], project=PROJECT)
        assert c.extra_args == extra
        c = m.Config(["test.py", *extra, "--files", "a", "b"], project=PROJECT)
        assert c.extra_args == extra

    def test_cov_args(self) -> None:
        cov_args = ["--cov-args", "run -a"]
        c = m.Config(["test.py", *cov_args], project=PROJECT)
        assert c.other.cov_args == "run -a"

    def test_multi_ranks_bad_launcher(self) -> None:
        msg = (
            "Requested multi-rank configuration with --ranks-per-node 4 but "
            "did not specify a launcher. Must use --launcher to specify a "
            "launcher."
        )
        with pytest.raises(RuntimeError, match=msg):
            m.Config(["test.py", "--ranks-per-node", "4"], project=PROJECT)

    def test_multi_nodes_bad_launcher(self) -> None:
        msg = (
            "Requested multi-node configuration with --nodes 4 but did not "
            "specify a launcher. Must use --launcher to specify a launcher."
        )
        with pytest.raises(RuntimeError, match=msg):
            m.Config(["test.py", "--nodes", "4"], project=PROJECT)

    @pytest.mark.parametrize("launch", ("mpirun", "jsrun", "srun"))
    def test_multi_ranks_good_launcher(self, launch: str) -> None:
        c = m.Config(
            ["test.py", "--ranks-per-node", "4", "--launcher", launch],
            project=PROJECT,
        )
        assert c.multi_node.launcher == launch
        assert c.multi_node.ranks_per_node == 4

    @pytest.mark.parametrize("launch", ("mpirun", "jsrun", "srun"))
    def test_multi_nodes_good_launcher(self, launch: str) -> None:
        c = m.Config(
            ["test.py", "--nodes", "4", "--launcher", launch], project=PROJECT
        )
        assert c.multi_node.launcher == launch
        assert c.multi_node.nodes == 4


class Test_test_files:
    # first two tests are too sensitive to actual repo state and run location

    @pytest.mark.skip
    def test_basic(self) -> None:
        c = m.Config(["test.py", "--root-dir", str(REPO_TOP)], project=PROJECT)

        assert len(c.test_files) > 0
        assert any("examples" in str(x) for x in c.test_files)
        assert any("integration" in str(x) for x in c.test_files)

    def test_error(self) -> None:
        c = m.Config(
            ["test.py", "--files", "a", "b", "--last-failed"], project=PROJECT
        )
        with pytest.raises(RuntimeError):
            _ = c.test_files

    @pytest.mark.parametrize("data", ("", " ", "\n", " \n "))
    def test_last_failed_empty(self, mocker: MockerFixture, data: str) -> None:
        mock_last_failed = mocker.mock_open(read_data=data)
        mocker.patch("builtins.open", mock_last_failed)
        c1 = m.Config(
            ["test.py", "--last-failed", "--root-dir", str(REPO_TOP)],
            project=PROJECT,
        )
        c2 = m.Config(
            ["test.py", "--root-dir", str(REPO_TOP)], project=PROJECT
        )
        assert c1.test_files == c2.test_files

    def test_last_failed(self, mocker: MockerFixture) -> None:
        mock_last_failed = mocker.mock_open(read_data="\nfoo\nbar\nbaz\n")
        mocker.patch("pathlib.Path.open", mock_last_failed)
        c = m.Config(
            ["test.py", "--last-failed", "--root-dir", str(REPO_TOP)],
            project=PROJECT,
        )
        assert c.test_files == (Path("foo"), Path("bar"), Path("baz"))
