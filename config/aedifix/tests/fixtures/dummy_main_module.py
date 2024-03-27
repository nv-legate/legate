# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.B
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

from collections.abc import Sequence
from os import environ
from pathlib import Path

from ...manager import ConfigurationManager
from .dummy_main_package import DummyMainPackage


class DummyMainModule(DummyMainPackage):
    def __init__(
        self, manager: ConfigurationManager, argv: Sequence[str]
    ) -> None:
        super().__init__(
            manager=manager,
            argv=argv,
            name="DummyMainModule",
            arch_name="AEDIFIX_PYTEST_ARCH",
            project_dir_name="AEDIFIX_PYTEST_DIR",
            project_dir_value=Path(environ["AEDIFIX_PYTEST_DIR"]),
        )

    @classmethod
    def from_argv(
        cls, manager: ConfigurationManager, argv: Sequence[str]
    ) -> DummyMainModule:
        return cls(manager, argv)
