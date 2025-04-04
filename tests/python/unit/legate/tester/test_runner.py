# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Consolidate test configuration from command-line and environment."""

from __future__ import annotations

from pathlib import Path

import pytest

import legate.tester.runner as m
from legate.tester.config import Config
from legate.tester.project import Project

PROJECT = Project()


class Runner:
    # TODO: update when legate/gtest command split happens
    def test_create(self) -> None:
        pass


class TestLegateRunner:
    def test_test_specs(self) -> None:
        c = Config(["test.py"], project=PROJECT)
        c.files = [Path("foo"), Path("bar")]
        r = m.LegateRunner()
        assert r.test_specs(c) == (
            m.TestSpec(Path("foo"), "foo"),
            m.TestSpec(Path("bar"), "bar"),
        )

    class Test_cmd_gdb:
        def test_multi_node_bad(self) -> None:
            c = Config(
                ["test.py", "--nodes", "2", "--launcher", "srun"],
                project=PROJECT,
            )
            r = m.LegateRunner()
            with pytest.raises(
                ValueError, match="--gdb can only be used with a single rank"
            ):
                r.cmd_gdb(c)

        def test_multi_rank_bad(self) -> None:
            c = Config(
                ["test.py", "--ranks-per-node", "2", "--launcher", "srun"],
                project=PROJECT,
            )
            r = m.LegateRunner()
            with pytest.raises(
                ValueError, match="--gdb can only be used with a single rank"
            ):
                r.cmd_gdb(c)

        def test_zero_tests_bad(self) -> None:
            c = Config(["test.py"], project=PROJECT)
            c.files = []
            r = m.LegateRunner()
            with pytest.raises(
                ValueError,
                match=(
                    r"--gdb can only be used with a single test "
                    r"\(none were given\)"
                ),
            ):
                r.cmd_gdb(c)

        def test_multi_tests_bad(self) -> None:
            c = Config(["test.py"], project=PROJECT)
            c.files = ["foo", "bar"]
            r = m.LegateRunner()
            with pytest.raises(
                ValueError, match="--gdb can only be used with a single test"
            ):
                r.cmd_gdb(c)

        def test_good(self) -> None:
            c = Config(["test.py"], project=PROJECT)
            c.files = ["foo"]
            r = m.LegateRunner()
            assert r.cmd_gdb(c) == r.cmd(m.TestSpec(Path("foo"), "foo"), c, [])

    class TestTestStage_cov_args:
        def test_without_cov_bin(self) -> None:
            c = Config(["test.py", "--cov-args", "run -a"], project=PROJECT)
            r = m.LegateRunner()
            assert r.cov_args(c) == []

        def test_with_cov_bin(self) -> None:
            cov_bin = "conda/envs/legate/bin/coverage"
            args = ["--cov-bin", cov_bin]
            c = Config(["test.py", *args], project=PROJECT)
            expected_result = [
                "--run-mode=python",
                cov_bin,
                *c.other.cov_args.split(),
            ]
            r = m.LegateRunner()
            assert r.cov_args(c) == expected_result

        def test_with_cov_bin_args_and_src_path(self) -> None:
            cov_bin = "conda/envs/legate/bin/coverage"
            cov_args = "run -a"
            cov_src_path = "source_path"
            args = [
                "--cov-bin",
                cov_bin,
                "--cov-args",
                cov_args,
                "--cov-src-path",
                cov_src_path,
            ]
            c = Config(["test.py", *args], project=PROJECT)
            expected_result = [
                "--run-mode=python",
                cov_bin,
                *cov_args.split(),
                "--source",
                cov_src_path,
            ]
            r = m.LegateRunner()
            assert r.cov_args(c) == expected_result


class TestGTestRunner:
    def test_test_specs(self) -> None:
        c = Config(["test.py"], project=PROJECT)
        c.gtest_tests = {Path("foo"): ["bar", "baz"]}
        r = m.GTestRunner()
        assert r.test_specs(c) == (
            m.TestSpec(Path("foo"), "bar", "bar"),
            m.TestSpec(Path("foo"), "baz", "baz"),
        )

    class Test_cmd:
        def test_cmd_single(self) -> None:
            c = Config(["test.py"], project=PROJECT)
            r = m.GTestRunner()
            args = (m.TestSpec(Path("foo"), "", "bar"), c, ["--custom"])
            assert r.cmd(*args) == r._cmd_single(*args)

        def test_cmd_multi_rank(self) -> None:
            c = Config(
                ["test.py", "--ranks-per-node", "2", "--launcher", "mpirun"],
                project=PROJECT,
            )
            r = m.GTestRunner()
            args = (m.TestSpec(Path("foo"), "", "bar"), c, ["--custom"])
            assert r.cmd(*args) == r._cmd_multi(*args)

        def test_cmd_multi_node_mpirun(self) -> None:
            # we do care about the launcher here, only mpirun will add args
            # that distinguish the result from _cmd_single
            c = Config(
                ["test.py", "--nodes", "2", "--launcher", "mpirun"],
                project=PROJECT,
            )
            r = m.GTestRunner()
            args = (m.TestSpec(Path("foo"), "", "bar"), c, ["--custom"])
            assert r.cmd(*args) == r._cmd_multi(*args)

        def test_cmd_multi_output_filename_mpirun(self) -> None:
            c = Config(
                [
                    "test.py",
                    "--nodes",
                    "2",
                    "--launcher",
                    "mpirun",
                    "--mpi-output-filename",
                    "a/b c/d.out",
                ],
                project=PROJECT,
            )
            r = m.GTestRunner()
            out = r.cmd(m.TestSpec(Path("foo"), "", "bar"), c, [])
            assert "--launcher-extra=--merge-stderr-to-stdout" in out
            assert '--launcher-extra="--output-filename"' in out
            assert "--launcher-extra='a/b c/d.out'" in out

        def test_cmd_multi_mpi_output_filename_non_mpirun(self) -> None:
            c = Config(
                [
                    "test.py",
                    "--nodes",
                    "2",
                    "--launcher",
                    "srun",
                    "--mpi-output-filename",
                    "a/b c/d.out",
                ],
                project=PROJECT,
            )
            r = m.GTestRunner()
            out = r.cmd(m.TestSpec(Path("foo"), "", "bar"), c, [])
            assert "--launcher-extra" not in out

    class Test_cmd_gdb:
        def test_multi_node_bad(self) -> None:
            c = Config(
                ["test.py", "--nodes", "2", "--launcher", "srun"],
                project=PROJECT,
            )
            r = m.GTestRunner()
            with pytest.raises(
                ValueError, match="--gdb can only be used with a single rank"
            ):
                r.cmd_gdb(c)

        def test_multi_rank_bad(self) -> None:
            c = Config(
                ["test.py", "--ranks-per-node", "2", "--launcher", "srun"],
                project=PROJECT,
            )
            r = m.GTestRunner()
            with pytest.raises(
                ValueError, match="--gdb can only be used with a single rank"
            ):
                r.cmd_gdb(c)

        def test_missing_gtest_file_bad(self) -> None:
            c = Config(["test.py"], project=PROJECT)
            c.gtest_tests = {Path(): []}
            r = m.GTestRunner()
            with pytest.raises(
                ValueError,
                match=(
                    r"--gdb can only be used with a single test "
                    r"\(none were given\)"
                ),
            ):
                r.cmd_gdb(c)

        def test_zero_tests_bad(self) -> None:
            c = Config(["test.py"], project=PROJECT)
            c.gtest_tests = {}
            r = m.GTestRunner()
            with pytest.raises(
                ValueError,
                match=(
                    r"--gdb can only be used with a single test "
                    r"\(none were given\)"
                ),
            ):
                r.cmd_gdb(c)

        def test_multi_tests_bad(self) -> None:
            c = Config(["test.py"], project=PROJECT)
            c.gtest_tests = {Path("foo"): ["bar", "baz"]}
            r = m.GTestRunner()
            with pytest.raises(
                ValueError, match="--gdb can only be used with a single test"
            ):
                r.cmd_gdb(c)

        def test_multi_tests_bad_multi(self) -> None:
            c = Config(["test.py"], project=PROJECT)
            c.gtest_tests = {Path("foo"): ["bar"], Path("baz"): ["bop"]}
            r = m.GTestRunner()
            with pytest.raises(
                ValueError, match="--gdb can only be used with a single test"
            ):
                r.cmd_gdb(c)

        def test_good(self) -> None:
            c = Config(["test.py"], project=PROJECT)
            c.gtest_tests = {Path("foo"): ["bar"]}
            r = m.GTestRunner()
            assert r.cmd_gdb(c) == r._cmd_single(
                m.TestSpec(Path("foo"), "", "bar"), c, []
            )

    class Test_gtest_args:
        def test_defaults(self) -> None:
            r = m.GTestRunner()
            assert r.gtest_args(m.TestSpec(Path("foo"), "", "bar")) == [
                "foo",
                "--gtest_filter=bar",
            ]

        def test_colors_enabled(self) -> None:
            r = m.GTestRunner()
            assert r.gtest_args(
                m.TestSpec(Path("foo"), "", "bar"), color=True
            ) == ["foo", "--gtest_filter=bar", "--gtest_color=yes"]

        def test_colors_disabled(self) -> None:
            r = m.GTestRunner()
            assert r.gtest_args(
                m.TestSpec(Path("foo"), "", "bar"), color=False
            ) == ["foo", "--gtest_filter=bar"]

        def test_gdb_enabled(self) -> None:
            r = m.GTestRunner()
            assert r.gtest_args(
                m.TestSpec(Path("foo"), "", "bar"), gdb=True
            ) == ["foo", "--gtest_filter=bar", "--gtest_catch_exceptions=0"]

        def test_gdb_disabled(self) -> None:
            r = m.GTestRunner()
            assert r.gtest_args(
                m.TestSpec(Path("foo"), "", "bar"), gdb=False
            ) == ["foo", "--gtest_filter=bar"]
