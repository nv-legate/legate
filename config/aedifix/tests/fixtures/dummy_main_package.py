# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.B
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from os import environ
from pathlib import Path

from ...package.main_package import MainPackage


class DummyMainPackage(MainPackage):
    @property
    def arch_value(self) -> str:
        return environ[self.arch_name]

    @property
    def project_dir_value(self) -> Path:
        return Path(environ[self.project_dir_name])
