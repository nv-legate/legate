# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dask.distributed import Client

# Constants
DEFAULT_SCHEDULER_PORT: Final[int] = 50000
DEFAULT_DASK_BASE_PORT: Final[int] = 50010


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


def _setenv(selfidx: int, peersaddr: Sequence[str]) -> None:
    r"""Set environment variables for worker communication.

    Parameters
    ----------
    selfidx : int
        Index of current worker in `peersaddr`.
    peersaddr : Sequence[str]
        Sequence of all worker addresses as strings in 'ip:port' format.
    """
    from legate.bootstrap import ucp_setup_peer  # noqa: PLC0415

    ucp_setup_peer(selfidx, peersaddr)


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

    peers = [w.addr for w in legate_worker_details.values()]

    for worker_idx, worker_id in enumerate(legate_worker_details.keys()):
        client.run(_setenv, worker_idx, peers, workers=[worker_id])


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
