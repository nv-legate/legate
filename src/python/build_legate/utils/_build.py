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

from typing import TYPE_CHECKING

from ._cmake_config import CMakeConfig
from ._io import vprint
from ._legate_config import get_legate_config
from ._utils import clean_skbuild_dir, fix_env, was_built_with_build_isolation

if TYPE_CHECKING:
    from ._types import BuildImpl, BuildKind, ConfigSettings


def prepare_config_settings(
    config_settings: ConfigSettings | None,
) -> ConfigSettings:
    r"""Create the config_settings dict, and fill it with configuration
    settings based on the C++ configure run.

    Parameters
    ----------
    config_settings : ConfigSettings
        The incoming config settings.

    Returns
    -------
    ConfigSettings
        The filled out config settings.
    """
    cmake_config = CMakeConfig()

    default_config_settings: ConfigSettings = {
        "build-dir": str(get_legate_config().SKBUILD_BUILD_DIR),
        "cmake.args": cmake_config.cmake_args,
        "cmake.build-type": cmake_config.build_type,
    }

    for name, define in cmake_config.cmake_defines.items():
        default_config_settings[f"cmake.define.{name}"] = define

    if config_settings is None:
        config_settings = {}

    # Don't use config_settings.update() because we want the user to be able to
    # override any of the default settings we put here.
    for key, value in default_config_settings.items():
        config_settings.setdefault(key, value)

    return config_settings


def build_impl(
    orig_impl: BuildImpl,
    build_kind: BuildKind,
    wheel_directory: str,
    config_settings: ConfigSettings | None = None,
    metadata_directory: str | None = None,
) -> str:
    r"""
    Wrap a scikit-build-core build function to execute with legate build
    system support.

    Parameters
    ----------
    orig_impl : BuildImpl
        The original scikit-build-core builder function.
    build_kind : BuildKind
        The kind of build to perform.
    wheel_directory : str
        The wheel directory to pass to the skbuild function.
    config_settings : ConfigSettings | None
        The command-line config settings.
    metadata_directory : str | None
        The directory containing package metadata (None if no metadata).

    Returns
    -------
    str
        The basename of the generated wheel file from scikit-build-core.
    """
    fix_env()

    if build_kind == "wheel" and was_built_with_build_isolation():
        # Explicitly uninstall legate if doing a clean/isolated build.
        #
        # A prior installation may have built and installed legate C++
        # dependencies.
        #
        # CMake will find and use them for the current build, which
        # would normally be correct, but pip uninstalls files from any
        # existing installation as the last step of the install
        # process, including the libraries found by CMake during the
        # current build.
        #
        # Therefore this uninstall step must occur *before* CMake
        # attempts to find these dependencies, triggering CMake to
        # build and install them again.
        clean_skbuild_dir()

    config_settings = prepare_config_settings(config_settings)

    vprint("using wheel directory", wheel_directory)
    vprint("using metadata directory", metadata_directory)
    vprint("using config_settings:")
    for key, value in config_settings.items():
        vprint(f"  {key} = {value}")

    return orig_impl(
        wheel_directory=wheel_directory,
        config_settings=config_settings,
        metadata_directory=metadata_directory,
    )
