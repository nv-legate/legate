# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import argparse

from legate.bootstrap import bootstrap_worker

SINGLE_CONTROLLER_CONFIG_KEY = "--single-controller-execution"
LEGATE_CONFIG_ENVVAR = "LEGATE_CONFIG"


def main(host: str, port: int) -> None:
    """Connect the worker process to an existing controller at host:port
    and wait for work to be assigned until runtime destruction for
    single controller execution.

    Parameters
    ----------
    host : str
        Controller host IP address
    port : int
        Controller port number that will accept worker connections.

    Raises
    ------
    ValueError
        If `--single-controller-execution` is not in the Legate configuration.
    """
    if SINGLE_CONTROLLER_CONFIG_KEY not in os.environ[LEGATE_CONFIG_ENVVAR]:
        msg = (
            f"Requires {SINGLE_CONTROLLER_CONFIG_KEY} in the "
            f"{LEGATE_CONFIG_ENVVAR} environment variable."
        )
        raise ValueError(msg)

    bootstrap_worker(host, port)

    # import legate.core to initialize the runtime
    import legate.core  # noqa: F401, PLC0415


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Start worker and connect to controller of Legate runtime."
    )
    parser.add_argument(
        "--host", type=str, required=True, help="Controller host address"
    )
    parser.add_argument(
        "--port", type=int, required=True, help="Controller port number"
    )
    args = parser.parse_args()

    main(**vars(args))
