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

import os
import sys
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from ..cmake.cmake_flags import CMakeString
from ..reconfigure import Reconfigure
from ..util.utility import subprocess_capture_output
from .fixtures.dummy_main_module import DummyMainModule

if TYPE_CHECKING:
    from .fixtures.dummy_manager import DummyManager


@pytest.fixture
def reconf(manager: DummyManager) -> Reconfigure:
    return Reconfigure(manager)


TEST_SANITIZED_ARGV_ARGS: tuple[
    tuple[tuple[str, ...], set[str], list[str], list[str]], ...
] = (
    (
        (
            "--arg_1",
            "value_1",
            "--arg2=value2",
            "--ephem_1",
            "--ephem_2=value2",
            "--ephem_3",
            "value3",
            "--arg-end",
        ),
        {"--ephem_1", "--ephem_2", "--ephem_3"},
        [],
        [
            "--AEDIFIX_PYTEST_ARCH={AEDIFIX_PYTEST_ARCH}",
            "--arg_1",
            "value_1",
            "--arg2=value2",
            "--arg-end",
        ],
    ),
    ((), set(), [], ["--AEDIFIX_PYTEST_ARCH={AEDIFIX_PYTEST_ARCH}"]),
    (
        ("--arg_1", "value_1", "--arg2=value2", "--arg-end"),
        set(),
        ["-DFOO=BAR", "-DBAZ='BOP BLIP'"],
        [
            "--AEDIFIX_PYTEST_ARCH={AEDIFIX_PYTEST_ARCH}",
            "--arg_1",
            "value_1",
            "--arg2=value2",
            "--arg-end",
            "--",
            "-DFOO=BAR",
            "-DBAZ='BOP BLIP'",
        ],
    ),
    (
        (
            "--arg_1",
            "value_1",
            "--arg2=value2",
            "--arg-ephem",
            "--arg-end",
            "--",
            "-DFOO=BAZ",
        ),
        {"--arg-ephem"},
        ["-DFOO=BAR", "-DBAZ='BOP BLIP'"],
        [
            "--AEDIFIX_PYTEST_ARCH={AEDIFIX_PYTEST_ARCH}",
            "--arg_1",
            "value_1",
            "--arg2=value2",
            "--arg-end",
            "--",
            "-DFOO=BAZ",
            "-DFOO=BAR",
            "-DBAZ='BOP BLIP'",
        ],
    ),
)


class TestReconfigure:
    def test_create(
        self,
        manager: DummyManager,
        AEDIFIX_PYTEST_DIR: Path,
        AEDIFIX_PYTEST_ARCH: str,
    ) -> None:
        reconf = Reconfigure(manager)
        assert isinstance(reconf.reconfigure_file, Path)
        assert (
            reconf.reconfigure_file.parent
            == AEDIFIX_PYTEST_DIR / AEDIFIX_PYTEST_ARCH
        )
        assert not reconf.reconfigure_file.parent.exists()
        assert not reconf.reconfigure_file.exists()

    def test_get_import_line(self) -> None:
        class DummyClass:
            pass

        mod_name, type_name = Reconfigure.get_import_line(DummyClass)
        assert mod_name == "config.aedifix.tests.test_reconfigure"
        assert type_name == "DummyClass"

    def ensure_reconfigure_file(
        self, reconfigure: Reconfigure, link_symlink: bool
    ) -> tuple[Path, Path]:
        # create project-dir/arch-name/reconfigure.py
        reconf_file = reconfigure.reconfigure_file
        reconf_file.parent.mkdir(exist_ok=False)
        reconf_text = f"foo, bar, baz: {random.random()}"
        reconf_file.write_text(reconf_text)
        # create the symlink
        # project-dir/reconfigure.py -> ./arch-name/reconfigure.py
        project_dir = reconfigure.project_dir
        symlink = project_dir / reconf_file.name
        assert not symlink.exists()
        if link_symlink:
            symlink.symlink_to(reconf_file.relative_to(project_dir))
        return reconf_file, symlink

    def test_backup_reconfigure_script_with_symlink(
        self, reconf: Reconfigure, AEDIFIX_PYTEST_DIR: Path
    ) -> None:
        reconf_file, symlink = self.ensure_reconfigure_file(
            reconfigure=reconf, link_symlink=True
        )
        reconf_text = reconf_file.read_text()

        reconf.backup_reconfigure_script()
        assert symlink.exists()
        assert symlink.is_symlink()
        assert (AEDIFIX_PYTEST_DIR / symlink.readlink()) == reconf_file
        assert reconf_file.exists()
        assert reconf_file.is_file()
        assert reconf_file.read_text() == reconf_text
        assert hasattr(reconf, "_backup")
        assert isinstance(reconf._backup, Path)
        assert reconf._backup.exists()
        assert reconf._backup.is_file()
        assert reconf._backup != reconf_file
        assert reconf._backup.read_text() == reconf_text

    def test_backup_reconfigure_script_without_symlink(
        self, reconf: Reconfigure
    ) -> None:
        reconf_file, symlink = self.ensure_reconfigure_file(
            reconfigure=reconf, link_symlink=False
        )
        reconf_text = reconf_file.read_text()

        reconf.backup_reconfigure_script()
        assert not symlink.exists()
        assert reconf_file.exists()
        assert reconf_file.is_file()
        assert reconf_file.read_text() == reconf_text
        assert hasattr(reconf, "_backup")
        assert reconf._backup is None

    @pytest.mark.parametrize(
        ("argv", "ephemeral_args", "extra_argv", "expected"),
        TEST_SANITIZED_ARGV_ARGS,
    )
    def test_sanitized_argv(
        self,
        reconf: Reconfigure,
        AEDIFIX_PYTEST_ARCH: str,
        argv: tuple[str, ...],
        ephemeral_args: set[str],
        extra_argv: list[str],
        expected: list[str],
    ) -> None:
        ret = reconf.sanitized_argv(argv, ephemeral_args, extra_argv)
        expected = [
            s.format(AEDIFIX_PYTEST_ARCH=AEDIFIX_PYTEST_ARCH) for s in expected
        ]
        assert ret == expected

    def test_finalize(
        self,
        reconf: Reconfigure,
        AEDIFIX_PYTEST_DIR: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        reconf_file = reconf.reconfigure_file
        assert not reconf_file.exists()
        reconf_file.parent.mkdir(exist_ok=False)

        def log_execute_command(cmd: list[Any], **kwargs: Any) -> None:
            subprocess_capture_output(list(map(str, cmd)))

        monkeypatch.setattr(
            reconf.manager, "log_execute_command", log_execute_command
        )
        reconf.manager.register_cmake_variable(CMakeString("CMAKE_COMMAND"))
        reconf.manager.set_cmake_variable("CMAKE_COMMAND", "cmake")

        reconf.finalize(DummyMainModule, set())

        assert reconf_file.exists()
        assert reconf_file.is_file()
        assert os.access(reconf_file, os.X_OK)
        text = reconf_file.read_text()
        assert "import DummyMainModule" in text
        assert "return basic_configure(tuple(argv), DummyMainModule)" in text

        symlink = AEDIFIX_PYTEST_DIR / reconf_file.name
        assert symlink.exists()
        assert symlink.is_symlink()
        assert symlink.parent / symlink.readlink() == reconf_file


if __name__ == "__main__":
    sys.exit(pytest.main())
