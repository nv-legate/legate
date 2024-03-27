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

from legate.tester import args as m, defaults


class TestParserDefaults:
    def test_features(self) -> None:
        assert m.parser.get_default("features") is None

    # -- selection

    def test_test_root(self) -> None:
        assert m.parser.get_default("test_root") is None

    def test_files(self) -> None:
        assert m.parser.get_default("files") is None

    def test_unit(self) -> None:
        assert m.parser.get_default("unit") is False

    def test_last_failed(self) -> None:
        assert m.parser.get_default("last_failed") is False

    # -- core

    def test_cpus(self) -> None:
        assert m.parser.get_default("cpus") == defaults.CPUS_PER_NODE

    def test_gpus(self) -> None:
        assert m.parser.get_default("gpus") == defaults.GPUS_PER_NODE

    def test_omps(self) -> None:
        assert m.parser.get_default("omps") == defaults.OMPS_PER_NODE

    def test_ompthreads(self) -> None:
        assert m.parser.get_default("ompthreads") == defaults.OMPTHREADS

    def test_utility(self) -> None:
        assert m.parser.get_default("utility") == 1

    # -- memory

    def test_sysmem(self) -> None:
        assert m.parser.get_default("sysmem") == defaults.SYS_MEMORY_BUDGET

    def test_fbmem(self) -> None:
        assert m.parser.get_default("fbmem") == defaults.GPU_MEMORY_BUDGET

    def test_numamem(self) -> None:
        assert m.parser.get_default("numamem") == defaults.NUMA_MEMORY_BUDGET

    # -- multi_node

    def test_nodes(self) -> None:
        assert m.parser.get_default("nodes") == defaults.NODES

    def test_ranks_per_node(self) -> None:
        assert (
            m.parser.get_default("ranks_per_node") == defaults.RANKS_PER_NODE
        )

    def test_launcher(self) -> None:
        assert m.parser.get_default("launcher") == "none"

    def test_launcher_extra(self) -> None:
        assert m.parser.get_default("launcher_extra") == []

    def test_mpi_output_filename(self) -> None:
        assert m.parser.get_default("mpi_output_filename") is None

    # -- execution

    def test_workers(self) -> None:
        assert m.parser.get_default("workers") is None

    def test_timeout(self) -> None:
        assert m.parser.get_default("timeout") is None

    def test_cpu_pin(self) -> None:
        assert m.parser.get_default("cpu_pin") == "partial"

    def test_gpu_delay(self) -> None:
        assert m.parser.get_default("gpu_delay") == defaults.GPU_DELAY

    def test_bloat_factor(self) -> None:
        assert (
            m.parser.get_default("bloat_factor") == defaults.GPU_BLOAT_FACTOR
        )

    # -- info

    def test_verbose(self) -> None:
        assert m.parser.get_default("verbose") == 0

    def test_debug(self) -> None:
        assert m.parser.get_default("debug") is False

    # -- other

    def test_legate_dir(self) -> None:
        assert m.parser.get_default("legate_dir") is None

    def test_gdb(self) -> None:
        assert m.parser.get_default("gdb") is False

    def test_cov_bin(self) -> None:
        assert m.parser.get_default("cov_bin") is None

    def test_cov_args(self) -> None:
        assert m.parser.get_default("cov_args") == "run -a --branch"

    def test_cov_src_path(self) -> None:
        assert m.parser.get_default("cov_src_path") is None

    def test_dry_run(self) -> None:
        assert m.parser.get_default("dry_run") is False

    def test_color(self) -> None:
        assert m.parser.get_default("color") is False


class TestParserConfig:
    def test_parser_epilog(self) -> None:
        assert (
            m.parser.epilog
            == "Any extra arguments will be forwarded to the Legate script"
        )

    def test_parser_description(self) -> None:
        assert m.parser.description == "Run the Legate test suite"
