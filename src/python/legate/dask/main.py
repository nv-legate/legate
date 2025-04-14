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

import sys
import argparse
from typing import Any

from dask.distributed import Client, LocalCluster

from .worker import DEFAULT_SCHEDULER_PORT, daskrun, setup_worker_env

__all__ = ("main",)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Create and return the argument parser for Legate Dask launcher.

    Returns
    -------
    argparse.Namespace
        A namespace containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Launch a Legate program on each worker of a Dask cluster"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "cmd", nargs=argparse.REMAINDER, help="Command to execute"
    )

    parser.add_argument(
        "--workers-per-node",
        type=int,
        default=1,
        help="Number of workers per node",
    )

    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    """Launch a Legate program on each worker of a dask cluster.

    This function sets up a Dask cluster (either local or connecting to an
    existing one) and executes the specified program across the workers. It
    handles cluster configuration and environment setup for Legate's
    peer-to-peer communication.

    Parameters
    ----------
    argv: list[str]
        Command line arguments, where the first argument is the program name to
        execute and the remaining args can include:
            * ``--workers-per-node=N``: Number of workers per node

    Returns
    -------
        0 on successful execution, non-zero on failure
    """
    args = _parse_args(argv[1:])
    cluster = LocalCluster(  # type: ignore[no-untyped-call]
        scheduler_port=DEFAULT_SCHEDULER_PORT, n_workers=args.workers_per_node
    )
    sched_addr = cluster.scheduler_address

    with cluster, Client(sched_addr) as client:  # type: ignore[no-untyped-call]
        setup_worker_env(client)

        # client.run returns a dict but pyright thinks it is a coroutine
        output: Any = client.run(daskrun, args.cmd)
        for w_output in output.values():
            sys.stdout.write(w_output)

    return 0
