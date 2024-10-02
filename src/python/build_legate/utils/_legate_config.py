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
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from typing import Final, Generator


@contextmanager
def _push_legate_dir_to_sys_path() -> Generator[None, None, None]:
    root_dir = str(Path(__file__).parents[4].resolve(strict=True))
    sys.path.insert(0, root_dir)

    try:
        yield
    finally:
        try:
            sys.path.remove(root_dir)
        except ValueError:
            # someone already removed it I guess
            pass


def _get_legate_dir() -> Path:
    if not (legate_dir := os.environ.get("LEGATE_DIR", "").strip()):
        with _push_legate_dir_to_sys_path():
            # Technically _push_legate_dir_to_sys_path() already gets us 99% of
            # the way to LEGATE_DIR, but let's use the get_legate_dir utility
            # anyway
            from scripts.get_legate_dir import get_legate_dir

        legate_dir = get_legate_dir()

    legate_path = Path(legate_dir).resolve(strict=True)
    os.environ["LEGATE_DIR"] = str(legate_path)
    return legate_path


def _get_legate_arch() -> str:
    if legate_arch := os.environ.get("LEGATE_ARCH", "").strip():
        return legate_arch

    try:
        with _push_legate_dir_to_sys_path():
            from scripts.get_legate_arch import (  # type: ignore[import-not-found,unused-ignore] # noqa: E501
                get_legate_arch,
            )
    except ModuleNotFoundError as mnfe:
        from ._io import ERROR_BANNER

        # User has not run configure yet
        raise RuntimeError(
            "\n"
            f"{ERROR_BANNER}"
            "\n"
            "Must export LEGATE_ARCH in environment before continuing "
            " and/or run configure"
            "\n"
            f"{ERROR_BANNER}"
        ) from mnfe

    legate_arch = get_legate_arch()
    os.environ["LEGATE_ARCH"] = legate_arch
    return legate_arch


class LegateConfig:
    def __init__(self) -> None:
        self.LEGATE_DIR: Final = _get_legate_dir()
        self.LEGATE_ARCH: Final = _get_legate_arch()
        self._sanity_check()

    @property
    def LEGATE_ARCH_DIR(self) -> Path:
        return self.LEGATE_DIR / self.LEGATE_ARCH

    @property
    def LEGATE_CMAKE_DIR(self) -> Path:
        return self.LEGATE_ARCH_DIR / "cmake_build"

    @property
    def SKBUILD_BUILD_DIR(self) -> Path:
        return self.LEGATE_ARCH_DIR / "skbuild_core"

    def _sanity_check(self) -> None:
        from ._io import ERROR_BANNER

        if (
            not self.LEGATE_ARCH_DIR.exists()
            or not self.LEGATE_CMAKE_DIR.exists()
        ):
            configure_cmd = (
                self.LEGATE_ARCH_DIR / f"reconfigure-{self.LEGATE_ARCH}.py"
            )
            if not configure_cmd.exists():
                configure_cmd = self.LEGATE_DIR / "configure"
            raise RuntimeError(
                "\n"
                f"{ERROR_BANNER}"
                "\n"
                f"Current Legate arch '{self.LEGATE_ARCH}' either does not"
                "\n"
                "exist, or does not appear to have been configured with python"
                "\n"
                "bindings enabled. Please run the following before continuing:"
                "\n"
                "\n"
                f"$ {configure_cmd} --LEGATE_ARCH='{self.LEGATE_ARCH}' "
                "--with-python"
                "\n"
                f"{ERROR_BANNER}"
                "\n"
            )


@cache
def get_legate_config() -> LegateConfig:
    return LegateConfig()
