# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys
import platform
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator
    from types import ModuleType

import pytest

from legate.bootstrap import (
    BOOTSTRAP_P2P_PLUGIN,
    REALM_UCP_BOOTSTRAP_MODE,
    WORKER_PEERS_INFO,
    WORKER_SELF_INFO,
    BootstrapMode,
    BootstrapPluginKind,
)
from legate.dask import main as daskexec
from legate.dask.worker import (
    DEFAULT_DASK_BASE_PORT,
    DEFAULT_SCHEDULER_PORT,
    _setenv,
    daskrun,
    setup_worker_env,
)

dask: ModuleType | None
try:
    from dask.distributed import Client, LocalCluster

    import dask
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


@pytest.mark.skipif(dask is None, reason="Dask is not installed")
class TestDaskWorker:
    @pytest.mark.parametrize("workers", range(1, 5))
    def test_workers(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        workers: int,
    ) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            ["daskrun", "--workers-per-node", str(workers), "hostname"],
        )
        daskexec()
        captured = capsys.readouterr()
        assert captured.out.strip().split("\n") == [platform.node()] * workers

    def test_daskrun(self) -> None:
        # for code coverage purposes
        stdout = daskrun(["hostname"])
        assert stdout.strip() == platform.node()

    @pytest.fixture
    def preserve_env(self) -> Generator[None]:
        env = os.environ.copy()
        yield
        os.environ.clear()
        os.environ.update(env)

    @pytest.mark.usefixtures("preserve_env")
    def test_set_env(self) -> None:
        # for code coverage purposes
        selfaddr = 0
        peersaddr = ["foo", "bar"]

        _setenv(selfaddr, peersaddr)
        assert os.environ[WORKER_SELF_INFO] == peersaddr[selfaddr]
        assert os.environ[WORKER_PEERS_INFO] == " ".join(peersaddr)
        assert os.environ[BOOTSTRAP_P2P_PLUGIN] == BootstrapPluginKind.UCP
        assert os.environ[REALM_UCP_BOOTSTRAP_MODE] == BootstrapMode.P2P


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
