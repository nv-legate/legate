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

import textwrap
from argparse import ArgumentParser
from pathlib import Path

_PACKAGE_DIR = Path(__file__).parent / "packages"
_PACKAGE_TEMPLATE = textwrap.dedent(
    """
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

from pathlib import Path
from typing import TYPE_CHECKING, Final

from ...cmake import CMAKE_VARIABLE, CMakePath
from ..package import Package

if TYPE_CHECKING:
    from ...util.argument_parser import ArgumentGroup
    from ...manager import ConfigurationManager


class {NAME}(Package):
    {NAME}_ROOT: Final = CMAKE_VARIABLE("{NAME}_ROOT", CMakePath)

    def __init__(self, manager: ConfigurationManager) -> None:
        r\"""Construct a {NAME} package.

        Parameters
        ----------
        manager : ConfigurationManager
            The configuration manager to manage this package.
        \"""
        super().__init__(manager=manager, name="{NAME}")

    def add_package_options(self, group: ArgumentGroup) -> None:
        r\"""Add Options for {NAME}.

        Parameters
        ----------
        group : ArgumentGroup
            The argument group to which the command line options
            should be added.
        \"""
        super().add_package_options(group)
        group.add_bool_argument(
            "--with-{NAME_LOWER}",
            help="Build with {NAME} support.",
        )
        group.add_argument(
            "--with-{NAME_LOWER}-dir",
            dest="{NAME_LOWER}_dir",
            type=Path,
            help="Path to {NAME} installation directory.",
        )

    def declare_dependencies(self) -> None:
        r\"""Declare Dependencies for {NAME}.\"""
        super().declare_dependencies()

    def setup(self) -> None:
        r\"""Setup {NAME}.\"""
        super().setup()

    def find_package(self) -> None:
        r\"""Find {NAME}.\"""
        super().find_package()

    def configure(self) -> None:
        r\"""Configure {NAME}.\"""
        super().configure()
        if self.enabled:
            self.set_flag_if_user_set(self.{NAME}_ROOT, self.cl_args.{NAME_LOWER}_dir)

    def finalize(self) -> None:
        r\"""Finalize {NAME}.\"""
        super().finalize()

    def summarize(self) -> str:
        r\"""Summarize {NAME}.

        Returns
        -------
        summary : str
            A summary string which describes the installation of {NAME}.
        \"""
        return super().summarize()


def create_package(manager: ConfigurationManager) -> {NAME}:
    return {NAME}(manager)
    """  # noqa E501
).strip()


def emit_template(path: Path, name: str) -> None:
    filled_template = _PACKAGE_TEMPLATE.format(
        NAME=name, NAME_LOWER=path.stem.casefold()
    )
    path.write_text(filled_template)
    print(f"Created new package module {name}: {path}")


def path_from_name(name: str) -> tuple[Path, str]:
    path = (_PACKAGE_DIR / name.casefold()).with_suffix(".py")
    if path.exists():
        raise RuntimeError(f"Package {name} already exists: {path}")
    return path, name


def main() -> None:
    parser = ArgumentParser(
        description="A barebones template for creating a new package"
    )
    parser.add_argument(
        "names",
        nargs="+",
        help="Names of the packages you would like to create",
    )
    args = parser.parse_args()
    paths = list(map(path_from_name, args.names))
    for path, name in paths:
        emit_template(path, name)


if __name__ == "__main__":
    main()
