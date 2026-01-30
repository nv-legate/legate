# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import socket

from .ucp import ucp_setup_peer


def bootstrap_world(world_size: int, host: str, port: int) -> None:
    """Listens for connections from n-1 workers, then sends all workers the
    full list of socket addresses for bootstrap configuration. Additionally,
    after coordination, this call sets up caller for bootstrap.

    Parameters
    ----------
    world_size : int
        The size of the world the caller is constructing. Should be the number
        workers intended plus 1 for the calling coordinator.
    host: str
        The host IP of the socket address the coordinator will listen on.
    port: int
        The port number of the socket address the coordinator will listen on.

    Notes
    -----
    1. Must be called before starting Legate Runtime via `import legate.core`.
    2. Must be called prior to calling `bootstrap_worker` on any workers.
    """
    clients = [f"{host}:{port}"]
    connections = []

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((host, port))
        server_sock.listen(world_size - 1)

        # Accept connections but don't close them until addresses are sent
        while len(clients) < world_size:
            conn, addr = server_sock.accept()
            # Keep conn open, don't "with conn"
            worker_addr = f"{addr[0]}:{addr[1]}"
            clients.append(worker_addr)
            connections.append(conn)  # Save connection

        # Now, send the full list to all clients
        all_clients_encoded = json.dumps(clients).encode("utf-8")
        for i, conn in enumerate(connections):
            conn.sendall((i + 1).to_bytes(4, "big") + all_clients_encoded)

    # setup coordinator for bootstrap via UCP
    ucp_setup_peer(0, clients)


def bootstrap_worker(coordinator_host: str, coordinator_port: int) -> None:
    """Connect to the coordinator that called `bootstrap_world` at host:port
    and receive the bootstrap information after all workers have connected
    to the coordinator. Then, setup the worker for bootstrap.

    Parameters
    ----------
    coordinator_host : str
        Coordinator host IP address
    coordinator_port : int
        Coordinator port number that will accept worker connections.

    Notes
    -----
    1. Must be called before starting Legate Runtime via `import legate.core`.
    2. Must be called after the coordinator has called `bootstrap_world`.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((coordinator_host, coordinator_port))

        # receive all the bytes associated with the rank and worker addresses
        data = b""
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            data += chunk

        # coordinator will send rank of worker in world
        # and socket addresses of all other workers
        rank = int.from_bytes(data[:4], "big")
        addr_list = json.loads(data[4:].decode("utf-8"))

    # setup worker for bootstrap via UCP
    ucp_setup_peer(rank, addr_list)
