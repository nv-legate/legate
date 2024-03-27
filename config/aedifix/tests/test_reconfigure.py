#!/usr/bin/env python3
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

import os
import random
import sys
import textwrap
from pathlib import Path

import pytest

from ..reconfigure import Reconfigure
from .fixtures.dummy_main_module import DummyMainModule
from .fixtures.dummy_manager import DummyManager


@pytest.fixture
def reconf(manager: DummyManager) -> Reconfigure:
    return Reconfigure(manager)


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
        "pre_create_symlink,link_symlink",
        # no point in doing (False, True) combination as link_symlink has no
        # effect then anyways
        ((True, True), (True, False), (False, False)),
    )
    def test_emit_file(
        self,
        reconf: Reconfigure,
        AEDIFIX_PYTEST_DIR: Path,
        pre_create_symlink: bool,
        link_symlink: bool,
    ) -> None:
        if pre_create_symlink:
            # This will create a reconfigure file with other text in it, let's
            # ensure that we are properly overwriting that.
            reconf_file, symlink = self.ensure_reconfigure_file(
                reconf, link_symlink=link_symlink
            )
            assert reconf_file.exists()
            if link_symlink:
                assert symlink.exists()
            else:
                assert not symlink.exists()
        else:
            reconf_file = reconf.reconfigure_file
            reconf_file.parent.mkdir(exist_ok=False)
            symlink = AEDIFIX_PYTEST_DIR / reconf_file.name
            assert not reconf_file.exists()
            assert not symlink.exists()

        text = r"""
        foo, bar, baz
        bop
        quux
        {random_number}
        """.format(
            random_number=random.random()
        )

        reconf.emit_file(text)

        assert reconf_file.exists()
        assert reconf_file.is_file()
        assert os.access(reconf_file, os.X_OK)
        assert reconf_file.read_text() == text

        assert symlink.exists()
        assert symlink.is_symlink()
        assert symlink.parent / symlink.readlink() == reconf_file

    def pre_test_finalize(
        self, reconf: Reconfigure, argv: tuple[str, ...]
    ) -> Path:
        assert isinstance(reconf.manager, DummyManager)
        reconf.manager.set_argv(argv)
        reconf_file = reconf.reconfigure_file
        assert not reconf_file.exists()
        reconf_file.parent.mkdir(exist_ok=False)
        return reconf_file

    def post_test_finalize(
        self, reconf_file: Path, symlink: Path, main_fn: str
    ) -> None:
        assert reconf_file.exists()
        assert reconf_file.is_file()
        assert os.access(reconf_file, os.X_OK)
        text = reconf_file.read_text()
        assert "import DummyMainModule" in text
        main_fn = textwrap.dedent(main_fn).strip()
        assert main_fn in text

        assert symlink.exists()
        assert symlink.is_symlink()
        assert symlink.parent / symlink.readlink() == reconf_file

    def test_finalize(
        self,
        reconf: Reconfigure,
        AEDIFIX_PYTEST_DIR: Path,
        AEDIFIX_PYTEST_ARCH: str,
    ) -> None:
        argv = (
            "--arg_1",
            "value_1",
            "--arg2=value2",
            "--ephem_1",
            "--ephem_2=value2",
            "--ephem_3",
            "value3",
            "--arg-end",
        )
        ephemeral_args = {"--ephem_1", "--ephem_2", "--ephem_3"}
        main_fn = r"""
        def main() -> int:
            argv = [
                "--AEDIFIX_PYTEST_ARCH={AEDIFIX_PYTEST_ARCH}",
                "--arg_1",
                "value_1",
                "--arg2=value2",
                "--arg-end",
            ] + sys.argv[1:]
            return basic_configure(
                tuple(argv), DummyMainModule
            )

        if __name__ == "__main__":
            sys.exit(main())
        """.format(
            AEDIFIX_PYTEST_ARCH=AEDIFIX_PYTEST_ARCH
        )

        reconf_file = self.pre_test_finalize(reconf=reconf, argv=argv)
        reconf.finalize(DummyMainModule, ephemeral_args)
        self.post_test_finalize(
            reconf_file=reconf_file,
            symlink=AEDIFIX_PYTEST_DIR / reconf_file.name,
            main_fn=main_fn,
        )

    def test_finalize_bare_argv(
        self,
        reconf: Reconfigure,
        AEDIFIX_PYTEST_DIR: Path,
        AEDIFIX_PYTEST_ARCH: str,
    ) -> None:
        argv: tuple[str, ...] = tuple()
        ephemeral_args: set[str] = set()
        main_fn = r"""
        def main() -> int:
            argv = [
                "--AEDIFIX_PYTEST_ARCH={AEDIFIX_PYTEST_ARCH}",
            ] + sys.argv[1:]
            return basic_configure(
                tuple(argv), DummyMainModule
            )

        if __name__ == "__main__":
            sys.exit(main())
        """.format(
            AEDIFIX_PYTEST_ARCH=AEDIFIX_PYTEST_ARCH
        )

        reconf_file = self.pre_test_finalize(reconf=reconf, argv=argv)
        reconf.finalize(DummyMainModule, ephemeral_args)
        self.post_test_finalize(
            reconf_file=reconf_file,
            symlink=AEDIFIX_PYTEST_DIR / reconf_file.name,
            main_fn=main_fn,
        )


if __name__ == "__main__":
    sys.exit(pytest.main())
