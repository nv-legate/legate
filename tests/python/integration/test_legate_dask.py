# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

import pytest

from legate.dask.worker import (
    BOOTSTRAP_P2P_PLUGIN,
    DEFAULT_DASK_BASE_PORT,
    DEFAULT_SCHEDULER_PORT,
    REALM_UCP_BOOTSTRAP_MODE,
    WORKER_PEERS_INFO,
    BootstrapMode,
    BootstrapPluginKind,
    setup_worker_env,
)

dask: ModuleType | None
try:
    import dask
    from dask.distributed import Client, LocalCluster
except ModuleNotFoundError:
    dask = None


@pytest.mark.skipif(dask is None, reason="Dask is not installed")
class TestDaskCluster:
    # This function is executed as Dask Task, hence the deferred import
    def getenv(self, env_var: str) -> str:
        import os  # noqa: PLC0415

        return os.environ[env_var]

    def test_legate_bootstrap_env(self) -> None:
        with (
            LocalCluster(  # type: ignore[no-untyped-call]
                scheduler_port=DEFAULT_SCHEDULER_PORT,
                n_workers=4,
                silence_logs=40,
            ) as cluster,
            Client(cluster) as client,  # type: ignore[no-untyped-call]
        ):
            setup_worker_env(client)
            peers_info = client.run(self.getenv, WORKER_PEERS_INFO)
            peers = next(iter(peers_info.values()))
            for peers_ids in peers_info.values():
                assert peers == peers_ids

            # Since the Dask cluster is local and all the workers are
            # processes on the same host, the IP of all workers is same.
            # Though The sorted list of ports of all the workers is strictly
            # increasing sequence starting with 50010.
            peers_info = next(iter(peers_info.values()))
            expected_port = DEFAULT_DASK_BASE_PORT
            for p in peers_info.split():
                ipport = p.split(":")
                port = int(ipport[1])
                assert expected_port == port
                expected_port += 1

            peers_plugin_name = client.run(self.getenv, BOOTSTRAP_P2P_PLUGIN)
            for plugin_name in peers_plugin_name.values():
                assert plugin_name == BootstrapPluginKind.UCP

            peers_bootstrap_mode = client.run(
                self.getenv, REALM_UCP_BOOTSTRAP_MODE
            )
            for bootstrap_mode in peers_bootstrap_mode.values():
                assert bootstrap_mode == BootstrapMode.P2P


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
