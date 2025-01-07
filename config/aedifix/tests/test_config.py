# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

import sys
from typing import TYPE_CHECKING

import pytest

from ..config import ConfigFile

if TYPE_CHECKING:
    from pathlib import Path

    from .fixtures.dummy_manager import DummyManager


@pytest.fixture
def config_file(manager: DummyManager, tmp_path: Path) -> ConfigFile:
    template = tmp_path / "foo.in"
    template.touch()
    return ConfigFile(manager=manager, config_file_template=template)


class TestConfigFile:
    def test_create(self, manager: DummyManager, tmp_path: Path) -> None:
        template = tmp_path / "foo.in"
        template.touch()
        config = ConfigFile(manager=manager, config_file_template=template)

        assert config.template_file.exists()
        assert config.template_file.is_file()
        assert config.template_file == template

        assert config._default_subst == {"PYTHON_EXECUTABLE": sys.executable}


if __name__ == "__main__":
    sys.exit(pytest.main())
