# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from .ucp_constants import (
    BOOTSTRAP_P2P_PLUGIN,
    REALM_UCP_BOOTSTRAP_MODE,
    WORKER_PEERS_INFO,
    WORKER_SELF_INFO,
    BootstrapMode,
    BootstrapPluginKind,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def ucp_setup_peer(my_rank: int, all_addrs: Sequence[str]) -> None:
    r"""Setup the peer for UCX bootstrap before Legate runtime initialization.

    Parameters
    ----------
    my_rank : int
        The index in `all_addrs` for the socket address of the calling worker.
    all_addrs : list[str]
        The socket addresses of all workers, each formatted as "ip:port".

    Raises
    ------
    IndexError
        If `my_rank` is not in [0, len(all_addrs)).

    Notes
    -----
    The following must be ensured when bootstrapping a peer via UCX:
    1. Legate must be configured and built with UCX support.
    2. All peers participating in the bootstrap must call this method.
    3. All peers that will be bootstrapped must have the same `all_addrs` list.
    4. All peers must select a unique `my_rank` in [0, len(all_addrs)).
    5. This function must be called prior to calling `import legate.core`.
    6. The environment variables mentioned below may no longer be modified.

    This function overrides the following environment variables on success:
    - `WORKER_SELF_INFO`
    - `WORKER_PEERS_INFO`
    - `BOOTSTRAP_P2P_PLUGIN`
    - `REALM_UCP_BOOTSTRAP_MODE`

    Environment variables are not modified when an exception is raised.

    If peer 1 had address "12.3.4.5" and wanted to bootstrap over
    socket 4567 and peer 2 had address "67.8.9.10" and wanted to bootstrap over
    socket 8910, then peer 1 would execute the following:

    ```python
    from legate.bootstrap import ucp_setup_peer
    ucp_setup_peer(0, ["12.3.4.5:4567", "67.8.9.10:8910"])
    ...
    import legate.core
    ```

    Similarly, peer 2 would execute the following:

    ```python
    from legate.bootstrap import ucp_setup_peer
    ucp_setup_peer(1, ["12.3.4.5:4567", "67.8.9.10:8910"])
    ...
    import legate.core
    ```
    """
    os.environ[WORKER_SELF_INFO] = all_addrs[my_rank]
    os.environ[WORKER_PEERS_INFO] = " ".join(all_addrs)
    os.environ[BOOTSTRAP_P2P_PLUGIN] = BootstrapPluginKind.UCP
    os.environ[REALM_UCP_BOOTSTRAP_MODE] = BootstrapMode.P2P
