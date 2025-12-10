# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator

import pytest

from legate.bootstrap import (
    BOOTSTRAP_P2P_PLUGIN,
    REALM_UCP_BOOTSTRAP_MODE,
    WORKER_PEERS_INFO,
    WORKER_SELF_INFO,
    BootstrapMode,
    BootstrapPluginKind,
    ucp_setup_peer,
)


class TestUcpSetupPeer:
    @pytest.fixture
    def preserve_env(self) -> Generator[None]:
        env = os.environ.copy()
        yield
        os.environ.clear()
        os.environ.update(env)

    @pytest.mark.usefixtures("preserve_env")
    def test_ucp_setup_peer(self) -> None:
        my_rank = 1
        all_addrs = ["127.0.0.1:12345", "127.0.0.1:12346"]

        ucp_setup_peer(my_rank, all_addrs)
        assert os.environ[WORKER_SELF_INFO] == all_addrs[my_rank]
        assert os.environ[WORKER_PEERS_INFO] == " ".join(all_addrs)
        assert os.environ[BOOTSTRAP_P2P_PLUGIN] == BootstrapPluginKind.UCP
        assert os.environ[REALM_UCP_BOOTSTRAP_MODE] == BootstrapMode.P2P

    @pytest.mark.usefixtures("preserve_env")
    def test_ucp_setup_peer_empty_addrs(self) -> None:
        my_rank = 1
        all_addrs: list[str] = []

        with pytest.raises(IndexError):
            ucp_setup_peer(my_rank, all_addrs)

    @pytest.mark.usefixtures("preserve_env")
    def test_ucp_setup_peer_invalid_idx(self) -> None:
        my_rank = 2
        all_addrs = ["127.0.0.1:12345", "127.0.0.1:12346"]

        with pytest.raises(IndexError):
            ucp_setup_peer(my_rank, all_addrs)


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
