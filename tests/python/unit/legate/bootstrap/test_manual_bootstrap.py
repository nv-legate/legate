# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys
import copy
import socket
import threading
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

import pytest

from legate.bootstrap import (
    BOOTSTRAP_P2P_PLUGIN,
    REALM_UCP_BOOTSTRAP_MODE,
    WORKER_PEERS_INFO,
    WORKER_SELF_INFO,
    BootstrapMode,
    BootstrapPluginKind,
    bootstrap_worker,
    bootstrap_world,
    manual,
)


@pytest.fixture
def preserve_env() -> Generator[None]:
    env = copy.deepcopy(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(env)


def _get_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as port_sock:
        port_sock.bind((host, 0))
        return port_sock.getsockname()[1]


def _make_socket_wrapper(
    original_socket: type[socket.socket],
    ready: threading.Event,
    accepted_conns: list[socket.socket],
) -> type:
    class SocketWrapper:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._sock = original_socket(*args, **kwargs)

        def __enter__(self) -> Self:
            self._sock.__enter__()
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: object | None,
        ) -> bool | None:
            self._sock.__exit__(exc_type, exc, tb)
            return None

        def listen(self, backlog: int) -> None:
            self._sock.listen(backlog)
            ready.set()

        def accept(self) -> tuple[socket.socket, tuple[str, int]]:
            conn, addr = self._sock.accept()
            accepted_conns.append(conn)
            return conn, addr

        def __getattr__(self, name: str) -> object:
            return getattr(self._sock, name)

    return SocketWrapper


def _make_ucp_setup_peer_wrapper(
    original_ucp_setup_peer: Callable[[int, list[str]], None],
    calls: list[tuple[str, int, list[str]]],
    coordinator_done: threading.Event,
) -> Callable[[int, list[str]], None]:
    def ucp_setup_peer_wrapper(my_rank: int, all_addrs: list[str]) -> None:
        calls.append(
            (threading.current_thread().name, my_rank, list(all_addrs))
        )
        if threading.current_thread().name == "worker":
            coordinator_done.wait(timeout=5)
        original_ucp_setup_peer(my_rank, all_addrs)
        if threading.current_thread().name == "coordinator":
            coordinator_done.set()

    return ucp_setup_peer_wrapper


def _run_bootstrap_world(
    host: str, port: int, errors: list[BaseException]
) -> None:
    try:
        bootstrap_world(2, host, port)
    except BaseException as exc:
        errors.append(exc)


def _run_bootstrap_worker(
    host: str, port: int, errors: list[BaseException]
) -> None:
    try:
        bootstrap_worker(host, port)
    except BaseException as exc:
        errors.append(exc)


def _run_bootstrap_threads(
    host: str, port: int, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[tuple[str, int, list[str]]], list[BaseException]]:
    ready = threading.Event()
    coordinator_done = threading.Event()
    calls: list[tuple[str, int, list[str]]] = []
    errors: list[BaseException] = []
    accepted_conns: list[socket.socket] = []

    socket_wrapper = _make_socket_wrapper(
        manual.socket.socket, ready, accepted_conns
    )
    ucp_setup_peer_wrapper = _make_ucp_setup_peer_wrapper(
        manual.ucp_setup_peer, calls, coordinator_done
    )

    monkeypatch.setattr(manual.socket, "socket", socket_wrapper)
    monkeypatch.setattr(manual, "ucp_setup_peer", ucp_setup_peer_wrapper)

    coordinator_thread = threading.Thread(
        target=_run_bootstrap_world,
        args=(host, port, errors),
        name="coordinator",
        daemon=True,
    )
    coordinator_thread.start()

    if not ready.wait(timeout=5):
        errors.append(TimeoutError("Coordinator failed to start"))

    worker_thread = threading.Thread(
        target=_run_bootstrap_worker,
        args=(host, port, errors),
        name="worker",
        daemon=True,
    )
    worker_thread.start()

    coordinator_thread.join(timeout=10)
    for conn in accepted_conns:
        with suppress(OSError):
            conn.close()
    worker_thread.join(timeout=10)

    if coordinator_thread.is_alive() or worker_thread.is_alive():
        errors.append(TimeoutError("Bootstrap threads did not finish"))

    return calls, errors


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

    def test_bootstrap_world_with_worker(
        self, preserve_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        host = "127.0.0.1"
        port = _get_free_port(host)
        calls, errors = _run_bootstrap_threads(host, port, monkeypatch)

        assert not errors
        assert len(calls) == 2
        call_map = {name: (rank, addrs) for name, rank, addrs in calls}
        coordinator_rank, coordinator_addrs = call_map["coordinator"]
        worker_rank, worker_addrs = call_map["worker"]

        assert coordinator_rank == 0
        assert worker_rank == 1
        assert coordinator_addrs == worker_addrs
        assert coordinator_addrs[0] == f"{host}:{port}"
        assert len(coordinator_addrs) == 2

        assert os.environ[WORKER_SELF_INFO] == worker_addrs[worker_rank]
        assert os.environ[WORKER_PEERS_INFO] == " ".join(worker_addrs)
        assert os.environ[BOOTSTRAP_P2P_PLUGIN] == BootstrapPluginKind.UCP
        assert os.environ[REALM_UCP_BOOTSTRAP_MODE] == BootstrapMode.P2P


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
