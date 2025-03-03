# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import re
import sys
from typing import TYPE_CHECKING

import pytest

from ..base import Configurable
from ..util.exception import WrongOrderError

if TYPE_CHECKING:
    from .fixtures.dummy_manager import DummyManager


@pytest.fixture
def configurable(manager: DummyManager) -> Configurable:
    return Configurable(manager)


class TestConfigurable:
    def test_create(self, manager: DummyManager) -> None:
        conf = Configurable(manager)
        assert conf.manager == manager
        assert conf.project_name == manager.project_name
        assert conf.project_arch == manager.project_arch
        assert conf.project_arch_name == manager.project_arch_name
        assert conf.project_dir == manager.project_dir
        assert conf.project_dir_name == manager.project_dir_name
        assert conf.project_arch_dir == manager.project_arch_dir
        assert conf.project_cmake_dir == manager.project_cmake_dir
        with pytest.raises(
            WrongOrderError, match=re.escape("Must call setup() first")
        ):
            _ = conf.cl_args


if __name__ == "__main__":
    sys.exit(pytest.main())
