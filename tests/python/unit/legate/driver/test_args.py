# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from argparse import SUPPRESS

import pytest

import legate.driver.args as m
from legate.util import defaults


class TestParserDefaults:
    def test_allow_abbrev(self) -> None:
        assert not m.parser.allow_abbrev

    # multi_node

    def test_nodes(self) -> None:
        assert m.parser.get_default("nodes") == defaults.LEGATE_NODES

    def test_ranks_per_node(self) -> None:
        assert (
            m.parser.get_default("ranks_per_node")
            == defaults.LEGATE_RANKS_PER_NODE
        )

    def test_launcher(self) -> None:
        assert m.parser.get_default("launcher") == "none"

    def test_launcher_extra(self) -> None:
        assert m.parser.get_default("launcher_extra") == []

    # binding

    def test_cpu_bind(self) -> None:
        assert m.parser.get_default("cpu_bind") is None

    def test_gpu_bind(self) -> None:
        assert m.parser.get_default("gpu_bind") is None

    def test_mem_bind(self) -> None:
        assert m.parser.get_default("mem_bind") is None

    def test_nic_bind(self) -> None:
        assert m.parser.get_default("nic_bind") is None

    # core

    def test_cpus(self) -> None:
        assert m.parser.get_default("cpus") is None

    def test_gpus(self) -> None:
        assert m.parser.get_default("gpus") is None

    def test_omps(self) -> None:
        assert m.parser.get_default("omps") is None

    def test_ompthreads(self) -> None:
        assert m.parser.get_default("ompthreads") is None

    def test_utility(self) -> None:
        assert m.parser.get_default("utility") is None

    # memory

    def test_sysmem(self) -> None:
        assert m.parser.get_default("sysmem") is None

    def test_numamem(self) -> None:
        assert m.parser.get_default("numamem") is None

    def test_fbmem(self) -> None:
        assert m.parser.get_default("fbmem") is None

    def test_zcmem(self) -> None:
        assert m.parser.get_default("zcmem") is None

    def test_regmem(self) -> None:
        assert m.parser.get_default("regmem") is None

    # profiling

    def test_profile(self) -> None:
        assert m.parser.get_default("profile") is False

    def test_provenance(self) -> None:
        assert m.parser.get_default("provenance") is None

    def test_cprofile(self) -> None:
        assert m.parser.get_default("cprofile") is False

    def test_nvprof(self) -> None:
        assert m.parser.get_default("nvprof") is False

    def test_nsys(self) -> None:
        assert m.parser.get_default("nsys") is False

    def test_nsys_extra(self) -> None:
        assert m.parser.get_default("nsys_extra") == []

    # logging

    def test_logging(self) -> None:
        assert m.parser.get_default("logging") is None

    def test_logdir(self) -> None:
        assert m.parser.get_default("logdir") == defaults.LEGATE_LOG_DIR

    def test_log_to_file(self) -> None:
        assert m.parser.get_default("log_to_file") is False

    # debugging

    def test_gdb(self) -> None:
        assert m.parser.get_default("gdb") is False

    def test_cuda_gdb(self) -> None:
        assert m.parser.get_default("cuda_gdb") is False

    def test_memcheck(self) -> None:
        assert m.parser.get_default("memcheck") is False

    def test_freeze_on_error(self) -> None:
        assert m.parser.get_default("freeze_on_error") is False

    def test_gasnet_trace(self) -> None:
        assert m.parser.get_default("gasnet_trace") is False

    # info

    def test_verbose(self) -> None:
        assert m.parser.get_default("verbose") is False

    def test_bind_detail(self) -> None:
        assert m.parser.get_default("bind_detail") is False

    # other

    def test_timing(self) -> None:
        assert m.parser.get_default("timing") is False

    def test_wrapper(self) -> None:
        assert m.parser.get_default("wrapper") == []

    def test_wrapper_inner(self) -> None:
        assert m.parser.get_default("wrapper_inner") == []

    def test_module(self) -> None:
        assert m.parser.get_default("module") is None

    def test_dry_run(self) -> None:
        assert m.parser.get_default("dry_run") is False

    def test_info(self) -> None:
        assert m.parser.get_default("info") == SUPPRESS


class TestParserConfig:
    def test_parser_epilog(self) -> None:
        assert m.parser.epilog is None

    def test_parser_description(self) -> None:
        assert m.parser.description == "Legate Driver"


class TestMultiNodeDefaults:
    def test_with_no_env(self) -> None:
        node_kw, ranks_per_node_kw = m.detect_multi_node_defaults()

        assert node_kw["default"] == defaults.LEGATE_NODES
        assert "auto-detected" not in node_kw["help"]

        assert ranks_per_node_kw["default"] == defaults.LEGATE_RANKS_PER_NODE
        assert "auto-detected" not in ranks_per_node_kw["help"]

    def test_with_OMPI(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "6")
        monkeypatch.setenv("OMPI_COMM_WORLD_LOCAL_SIZE", "2")

        node_kw, ranks_per_node_kw = m.detect_multi_node_defaults()

        assert node_kw["default"] == 3
        assert "OMPI" in node_kw["help"]

        assert ranks_per_node_kw["default"] == 2
        assert "OMPI" in ranks_per_node_kw["help"]

    def test_with_OMPI_incompatible(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "5")
        monkeypatch.setenv("OMPI_COMM_WORLD_LOCAL_SIZE", "3")

        with pytest.raises(ValueError) as e:  # noqa: PT011
            m.detect_multi_node_defaults()

        assert "OMPI_COMM_WORLD_SIZE=5" in str(e.value)
        assert "OMPI_COMM_WORLD_LOCAL_SIZE=3" in str(e.value)

    def test_with_OMPI_bad_world_size(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "5.2")
        monkeypatch.setenv("OMPI_COMM_WORLD_LOCAL_SIZE", "3")

        with pytest.raises(ValueError) as e:  # noqa: PT011
            m.detect_multi_node_defaults()
        assert "OMPI_COMM_WORLD_SIZE=5.2" in str(e.value)
        assert "OMPI_COMM_WORLD_LOCAL_SIZE=3" in str(e.value)

    def test_with_OMPI_bad_local_size(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "5")
        monkeypatch.setenv("OMPI_COMM_WORLD_LOCAL_SIZE", "3.2")

        with pytest.raises(ValueError) as e:  # noqa: PT011
            m.detect_multi_node_defaults()
        assert "OMPI_COMM_WORLD_SIZE=5" in str(e.value)
        assert "OMPI_COMM_WORLD_LOCAL_SIZE=3.2" in str(e.value)

    def test_with_MV2(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MV2_COMM_WORLD_SIZE", "6")
        monkeypatch.setenv("MV2_COMM_WORLD_LOCAL_SIZE", "2")

        node_kw, ranks_per_node_kw = m.detect_multi_node_defaults()

        assert node_kw["default"] == 3
        assert "MV2" in node_kw["help"]

        assert ranks_per_node_kw["default"] == 2
        assert "MV2" in ranks_per_node_kw["help"]

    def test_with_MV2_incompatible(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MV2_COMM_WORLD_SIZE", "5")
        monkeypatch.setenv("MV2_COMM_WORLD_LOCAL_SIZE", "3")

        with pytest.raises(ValueError) as e:  # noqa: PT011
            m.detect_multi_node_defaults()
        assert "MV2_COMM_WORLD_SIZE=5" in str(e.value)
        assert "MV2_COMM_WORLD_LOCAL_SIZE=3" in str(e.value)

    def test_with_MV2_bad_world_size(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MV2_COMM_WORLD_SIZE", "5.2w")
        monkeypatch.setenv("MV2_COMM_WORLD_LOCAL_SIZE", "3")

        with pytest.raises(ValueError) as e:  # noqa: PT011
            m.detect_multi_node_defaults()
        assert "MV2_COMM_WORLD_SIZE=5.2" in str(e.value)
        assert "MV2_COMM_WORLD_LOCAL_SIZE=3" in str(e.value)

    def test_with_MV2_bad_local_size(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MV2_COMM_WORLD_SIZE", "5")
        monkeypatch.setenv("MV2_COMM_WORLD_LOCAL_SIZE", "3.2")

        with pytest.raises(ValueError) as e:  # noqa: PT011
            m.detect_multi_node_defaults()
        assert "MV2_COMM_WORLD_SIZE=5" in str(e.value)
        assert "MV2_COMM_WORLD_LOCAL_SIZE=3.2" in str(e.value)

    def test_with_SLURM_with_integer_tasks_per_node(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SLURM_TASKS_PER_NODE", "3")
        monkeypatch.setenv("SLURM_JOB_NUM_NODES", "2")

        node_kw, ranks_per_node_kw = m.detect_multi_node_defaults()

        assert node_kw["default"] == 1
        assert "SLURM" in node_kw["help"]

        assert ranks_per_node_kw["default"] == 3
        assert "SLURM" in ranks_per_node_kw["help"]

    def test_with_SLURM_with_combined_tasks_per_node(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SLURM_TASKS_PER_NODE", "3(x2)")
        monkeypatch.setenv("SLURM_JOB_NUM_NODES", "2")

        node_kw, ranks_per_node_kw = m.detect_multi_node_defaults()

        assert node_kw["default"] == 2
        assert "SLURM" in node_kw["help"]

        assert ranks_per_node_kw["default"] == 3
        assert "SLURM" in ranks_per_node_kw["help"]

    def test_with_SLURM_with_bad_tasks_per_node(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SLURM_TASKS_PER_NODE", "3(x2),10")
        monkeypatch.setenv("SLURM_JOB_NUM_NODES", "2")

        with pytest.raises(ValueError) as e:  # noqa: PT011
            m.detect_multi_node_defaults()
        assert "SLURM_TASKS_PER_NODE=3(x2),10" in str(e.value)

    def test_with_SLURM_with_ntasks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SLURM_NTASKS", "6")
        monkeypatch.setenv("SLURM_JOB_NUM_NODES", "2")

        node_kw, ranks_per_node_kw = m.detect_multi_node_defaults()

        assert node_kw["default"] == 2
        assert "SLURM" in node_kw["help"]

        assert ranks_per_node_kw["default"] == 3
        assert "SLURM" in ranks_per_node_kw["help"]

    def test_with_SLURM_with_nprocs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SLURM_NPROCS", "6")
        monkeypatch.setenv("SLURM_JOB_NUM_NODES", "2")

        node_kw, ranks_per_node_kw = m.detect_multi_node_defaults()

        assert node_kw["default"] == 2
        assert "SLURM" in node_kw["help"]

        assert ranks_per_node_kw["default"] == 3
        assert "SLURM" in ranks_per_node_kw["help"]

    def test_with_SLURM_prefers_ntasks_over_procs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SLURM_PROCS", "80")
        monkeypatch.setenv("SLURM_NTASKS", "6")
        monkeypatch.setenv("SLURM_JOB_NUM_NODES", "2")

        node_kw, ranks_per_node_kw = m.detect_multi_node_defaults()

        assert node_kw["default"] == 2
        assert "SLURM" in node_kw["help"]

        assert ranks_per_node_kw["default"] == 3
        assert "SLURM" in ranks_per_node_kw["help"]

    def test_with_SLURM_incompatible_num_nodes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SLURM_NTASKS", "5")
        monkeypatch.setenv("SLURM_JOB_NUM_NODES", "3")

        with pytest.raises(ValueError) as e:  # noqa: PT011
            m.detect_multi_node_defaults()
        assert "SLURM_NTASKS=5" in str(e.value)
        assert "SLURM_JOB_NUM_NODES=3" in str(e.value)

    def test_with_SLURM_bad_num_nodes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SLURM_NTASKS", "5")
        monkeypatch.setenv("SLURM_JOB_NUM_NODES", "3.2")

        with pytest.raises(ValueError) as e:  # noqa: PT011
            m.detect_multi_node_defaults()
        assert "SLURM_NTASKS=5" in str(e.value)
        assert "SLURM_JOB_NUM_NODES=3.2" in str(e.value)

    def test_with_SLURM_bad_ntasks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SLURM_NTASKS", "5.2")
        monkeypatch.setenv("SLURM_JOB_NUM_NODES", "3")

        with pytest.raises(ValueError) as e:  # noqa: PT011
            m.detect_multi_node_defaults()
        assert "SLURM_NTASKS=5.2" in str(e.value)
        assert "SLURM_JOB_NUM_NODES=3" in str(e.value)

    def test_with_SLURM_incompatible_ntasks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SLURM_NTASKS", "5")
        monkeypatch.setenv("SLURM_JOB_NUM_NODES", "3")

        with pytest.raises(ValueError) as e:  # noqa: PT011
            m.detect_multi_node_defaults()
        assert "SLURM_NTASKS=5" in str(e.value)
        assert "SLURM_JOB_NUM_NODES=3" in str(e.value)

    def test_with_SLURM_bad_nprocs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SLURM_NPROCS", "5.2")
        monkeypatch.setenv("SLURM_JOB_NUM_NODES", "3")

        with pytest.raises(ValueError) as e:  # noqa: PT011
            m.detect_multi_node_defaults()
        assert "SLURM_NPROCS=5.2" in str(e.value)
        assert "SLURM_JOB_NUM_NODES=3" in str(e.value)

    def test_with_SLURM_incompatible_nprocs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SLURM_NPROCS", "5")
        monkeypatch.setenv("SLURM_JOB_NUM_NODES", "3")

        with pytest.raises(ValueError) as e:  # noqa: PT011
            m.detect_multi_node_defaults()
        assert "SLURM_NPROCS=5" in str(e.value)
        assert "SLURM_JOB_NUM_NODES=3" in str(e.value)

    # test same as no_env -- auto-detect for PMI is unsupported
    def test_with_PMI(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PMI_SIZE", "6")
        node_kw, ranks_per_node_kw = m.detect_multi_node_defaults()

        assert node_kw["default"] == defaults.LEGATE_NODES
        assert "auto-detected" not in node_kw["help"]

        assert ranks_per_node_kw["default"] == defaults.LEGATE_RANKS_PER_NODE
        assert "auto-detected" not in ranks_per_node_kw["help"]
