# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys
import copy
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
    bootstrap_world,
)


@pytest.fixture
def preserve_env() -> Generator[None]:
    env = copy.deepcopy(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(env)


class TestBootstrapWorld:
    def test_bootstrap_self(self, preserve_env: None) -> None:
        host = "127.0.0.1"
        port = 55505
        socket_addr = f"{host}:{port}"

        bootstrap_world(1, host, port)

        assert os.environ[WORKER_SELF_INFO] == socket_addr
        assert os.environ[WORKER_PEERS_INFO] == socket_addr
        assert os.environ[BOOTSTRAP_P2P_PLUGIN] == BootstrapPluginKind.UCP
        assert os.environ[REALM_UCP_BOOTSTRAP_MODE] == BootstrapMode.P2P


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
