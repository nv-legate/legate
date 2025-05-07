# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from importlib import reload
from typing import TYPE_CHECKING, Any

import pytest

from legate.driver import Config, Launcher
from legate.driver.config import MultiNode
from legate.util.system import System

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from types import ModuleType

    from .util import GenConfig, GenSystem


@pytest.fixture
def clear_and_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[str, ModuleType], None]:
    def _inner(name_or_names: str | Iterable[str], m: ModuleType) -> None:
        if isinstance(name_or_names, str):
            name_or_names = [name_or_names]

        for name in name_or_names:
            monkeypatch.delenv(name, raising=False)

        reload(m)

    return _inner


@pytest.fixture
def set_and_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[str, str, ModuleType], None]:
    def _inner(name: str, value: str, m: ModuleType) -> None:
        monkeypatch.setenv(name, value)
        reload(m)

    return _inner


@pytest.fixture
def genconfig() -> Any:
    def _config(
        args: list[str] | None = None,
        *,
        fake_module: str | None = "foo.py",
        multi_rank: tuple[int, int] | None = None,
    ) -> Config:
        args = ["legate"] + (args or [])
        if fake_module:
            args += [fake_module]

        config = Config(args)

        if multi_rank:
            # This is annoying but we can only replace the entire dataclass
            launcher = config.multi_node.launcher
            launcher_extra = config.multi_node.launcher_extra
            config.multi_node = MultiNode(
                *multi_rank, launcher, launcher_extra
            )

        return config

    return _config


@pytest.fixture
def gensystem(monkeypatch: pytest.MonkeyPatch) -> Any:
    def _system(
        rank_env: dict[str, str] | None = None, os: str | None = None
    ) -> System:
        if rank_env:
            for k, v in rank_env.items():
                monkeypatch.setenv(k, v)
        system = System()
        if os:
            monkeypatch.setattr(system, "os", os)
        return system

    return _system


@pytest.fixture
def genobjs(
    genconfig: GenConfig, gensystem: GenSystem, monkeypatch: pytest.MonkeyPatch
) -> Any:
    def _objs(
        args: list[str] | None = None,
        *,
        fake_module: str | None = "foo.py",
        multi_rank: tuple[int, int] | None = None,
        rank_env: dict[str, str] | None = None,
        os: str | None = None,
    ) -> tuple[Config, System, Launcher]:
        config = genconfig(
            args, fake_module=fake_module, multi_rank=multi_rank
        )
        system = gensystem(rank_env, os)
        launcher = Launcher.create(config, system)
        return config, system, launcher

    return _objs
