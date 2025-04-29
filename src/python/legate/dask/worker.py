# Copyright 2025-2025 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from dask.distributed import Client

# Constants
DEFAULT_SCHEDULER_PORT: Final[int] = 50000
DEFAULT_DASK_BASE_PORT: Final[int] = 50010

# Environment variable constants
WORKER_PEERS_INFO = "WORKER_PEERS_INFO"
WORKER_SELF_INFO = "WORKER_SELF_INFO"
BOOTSTRAP_P2P_PLUGIN = "BOOTSTRAP_P2P_PLUGIN"
REALM_UCP_BOOTSTRAP_MODE = "REALM_UCP_BOOTSTRAP_MODE"


class BootstrapPluginKind(str, Enum):
    UCP = "realm_ucp_bootstrap_p2p.so"


class BootstrapMode(str, Enum):
    P2P = "p2p"


@dataclass(frozen=True)
class WorkerDetails:
    ip: str
    port: int

    @property
    def addr(self) -> str:
        r"""Return the worker's address.

        return: The worker's address in 'ip:port' format.
        rtype: str
        """
        return f"{self.ip}:{self.port}"


def _setenv(selfaddr: str, peersaddr: str) -> None:
    r"""Set environment variables for worker communication.

    Parameters
    ----------
    selfaddr : str
        The address (ip:port) of the current worker.
    peersaddr : str
        Space-separated string of all worker addresses.
    """
    import os

    os.environ[WORKER_SELF_INFO] = selfaddr
    os.environ[WORKER_PEERS_INFO] = peersaddr
    os.environ[BOOTSTRAP_P2P_PLUGIN] = BootstrapPluginKind.UCP
    os.environ[REALM_UCP_BOOTSTRAP_MODE] = BootstrapMode.P2P


def setup_worker_env(client: Client) -> None:
    r"""Set up the Legate environment for each Dask worker.

    Parameters
    ----------
    client : Client
        A Dask Client instance connected to the cluster.
    """
    workers = client.scheduler_info()["workers"]
    legate_worker_details: dict[str, WorkerDetails] = {}
    uniq_port: dict[str, int] = {}

    for worker in workers:
        addr = worker.removeprefix("tcp://")
        ip, port = addr.split(":")

        try:
            uniq_port[ip] += 1
        except KeyError:
            uniq_port[ip] = DEFAULT_DASK_BASE_PORT

        port = uniq_port[ip]
        legate_worker_details[worker] = WorkerDetails(ip, port)

    peers = " ".join(w.addr for w in legate_worker_details.values())

    for worker_id, worker_detail in legate_worker_details.items():
        client.run(_setenv, worker_detail.addr, peers, workers=[worker_id])


def daskrun(cmd: list[str]) -> str:
    r"""Execute a program in the current environment.

    Parameters
    ----------
    cmd : list[str]
        The command to execute.

    Returns
    -------
    str
        The program's stdout and stderr output
    """
    return subprocess.run(
        cmd,
        text=True,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout
