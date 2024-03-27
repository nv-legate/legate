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

import importlib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..package import Package
    from ...manager import ConfigurationManager


def load_packages(manager: ConfigurationManager) -> list[Package]:
    r"""Load all package modules in the packages directory, and return the
    constructed packages.

    Parameters
    ----------
    manager : ConfigurationManager
        The configuration manager with which to construct the packages.

    Returns
    -------
    packages : list[Package]
        The list of loaded packages.
    """
    assert __package__, "Cannot auto-load packages without relative imports!"
    packages = []
    cur_dir = Path(__file__).parent
    manager.log(f"Using package directory: {cur_dir}")
    for module_file in cur_dir.iterdir():
        manager.log(f"Attempting to load package: {module_file}")
        if module_file.is_dir() or module_file.name.startswith("_"):
            # skip __init__.py, __pycache__ and any other directories
            manager.log(
                f"Skipping loading package: {module_file} is not a package!"
            )
            continue
        module = importlib.import_module(f".{module_file.stem}", __package__)
        manager.log(f"Loaded package: {module}")
        conf_obj = module.create_package(manager)
        manager.log(f"Adding package: {conf_obj}")
        packages.append(conf_obj)
    return packages
