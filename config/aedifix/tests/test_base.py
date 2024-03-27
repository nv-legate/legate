# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
from __future__ import annotations

import re
import sys

import pytest

from ..base import Configurable
from ..util.exception import WrongOrderError
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
            conf.cl_args


if __name__ == "__main__":
    sys.exit(pytest.main())
