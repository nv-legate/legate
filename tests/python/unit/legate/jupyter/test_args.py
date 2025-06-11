# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import legate.driver.args as m
from legate.util import defaults


class TestParserDefaults:
    def test_allow_abbrev(self) -> None:
        assert not m.parser.allow_abbrev

    # kernel

    def test_no_user(self) -> None:
        assert m.parser.get_default("user") is None

    def test_name(self) -> None:
        assert m.parser.get_default("name") is None

    def test_display_name(self) -> None:
        assert m.parser.get_default("display_name") is None

    def test_prefix(self) -> None:
        assert m.parser.get_default("prefix") is None

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

    # info

    def test_verbose(self) -> None:
        assert m.parser.get_default("verbose") is False
