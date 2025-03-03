# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.B
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import random
import string
from os import environ
from pathlib import Path

import pytest

from .fixtures.dummy_main_package import DummyMainPackage  # noqa: F401
from .fixtures.dummy_manager import DummyManager, manager  # noqa: F401


def _id_generator(
    size: int = 8, chars: str = string.ascii_uppercase + string.digits
) -> str:
    return "".join(random.choice(chars) for _ in range(size)).casefold()


@pytest.fixture(scope="session", autouse=True)
def setup_env() -> None:
    environ["__AEDIFIX_TESTING_DO_NOT_USE_OR_YOU_WILL_BE_FIRED__"] = "1"


@pytest.fixture(autouse=True)
def setup_project_dir(tmp_path_factory: pytest.TempPathFactory) -> None:
    tmp_path = tmp_path_factory.mktemp(_id_generator(size=16))
    environ["AEDIFIX_PYTEST_DIR"] = str(tmp_path)
    print("\nAEDIFIX_PYTEST_DIR =", tmp_path)  # noqa: T201


@pytest.fixture(scope="session", autouse=True)
def setup_project_arch() -> None:
    arch_val = "arch-pytest"
    environ["AEDIFIX_PYTEST_ARCH"] = arch_val
    print("AEDIFIX_PYTEST_ARCH =", arch_val)  # noqa: T201


@pytest.fixture
def AEDIFIX_PYTEST_DIR() -> Path:
    return Path(environ["AEDIFIX_PYTEST_DIR"])


@pytest.fixture
def AEDIFIX_PYTEST_ARCH() -> str:
    return environ["AEDIFIX_PYTEST_ARCH"]
