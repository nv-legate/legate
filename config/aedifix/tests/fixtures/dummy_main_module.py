# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.B
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from os import environ
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Final

from .dummy_main_package import DummyMainPackage

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...manager import ConfigurationManager

_tmp_file: Final = NamedTemporaryFile()  # noqa: SIM115
_tmp_path: Final = Path(_tmp_file.name)


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
            project_config_file_template=_tmp_path,
        )

    @classmethod
    def from_argv(
        cls, manager: ConfigurationManager, argv: Sequence[str]
    ) -> DummyMainModule:
        return cls(manager, argv)
